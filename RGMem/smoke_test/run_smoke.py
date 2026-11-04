"""Isolated RGMem smoke-test runner.

Nothing runs on import. Mutating/API stages require an explicit confirmation flag.
The runner does not modify the official scripts; it injects their blank path
settings only inside the current Python process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path


SMOKE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SMOKE_ROOT.parent
FIXTURE_PATH = SMOKE_ROOT / "fixtures" / "minimal_dialogue.jsonl"
ARTIFACTS_ROOT = SMOKE_ROOT / "artifacts"
ROLE_ID = 91001
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _settings():
    from config import settings

    return settings


def _require_confirmation(args: argparse.Namespace, flag_name: str) -> None:
    if not getattr(args, flag_name):
        cli_flag = "--" + flag_name.replace("_", "-")
        raise SystemExit(
            f"Refusing to continue without {cli_flag}. "
            "Review smoke_test/README.md before enabling writes or API calls."
        )


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SystemExit(
            "run-id must contain only letters, digits, underscores, or hyphens "
            "and be at most 64 characters."
        )
    return run_id


def _run_directory(run_id: str) -> Path:
    return ARTIFACTS_ROOT / _validate_run_id(run_id)


def _manifest_path(run_id: str) -> Path:
    return _run_directory(run_id) / "manifest.json"


def _load_manifest(run_id: str) -> dict:
    path = _manifest_path(run_id)
    if not path.is_file():
        raise SystemExit(f"Missing smoke manifest: {path}. Run the l0 stage first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_stage(run_id: str, stage: str) -> None:
    marker = _run_directory(run_id) / f".{stage}.started"
    try:
        with marker.open("x", encoding="utf-8") as file:
            file.write(datetime.now(timezone.utc).isoformat())
    except FileExistsError:
        raise SystemExit(
            f"Stage '{stage}' has already been started for run-id '{run_id}'. "
            "Use a new run-id instead of replaying a mutating stage."
        )


def _elasticsearch_client():
    from elasticsearch import Elasticsearch

    settings = _settings()
    return Elasticsearch(
        settings.ES_CONFIG["host"],
        basic_auth=(settings.ES_CONFIG["user"], settings.ES_CONFIG["password"]),
    )


def stage_preflight(_: argparse.Namespace) -> None:
    """Read-only connectivity and configuration inspection."""
    import redis
    from neo4j import GraphDatabase

    settings = _settings()
    es = _elasticsearch_client()
    redis_client = redis.Redis(**settings.REDIS_CONFIG)
    graph_config = settings.L1_GRAPH_STORE_CONFIG["config"]
    driver = GraphDatabase.driver(
        graph_config["url"],
        auth=(graph_config["username"], graph_config["password"]),
    )
    try:
        driver.verify_connectivity()
        print("Elasticsearch reachable:", es.ping())
        print("L0 index:", settings.ES_INDEX_TOPICAL_L0)
        print(
            "L0 index exists:",
            es.indices.exists(index=settings.ES_INDEX_TOPICAL_L0),
        )
        print("Redis reachable:", redis_client.ping())
        print("Neo4j reachable: True")
        print("L0 LLM provider:", settings.LLM_PROVIDER)
        print("L0 embedding provider:", settings.EMBEDDING_PROVIDER)
        print(
            "L0 embedding dimensions:",
            settings.EMBEDDING_CONFIGS[settings.EMBEDDING_PROVIDER]["dimensions"],
        )
    finally:
        driver.close()


def stage_init_es(args: argparse.Namespace) -> None:
    """Create the documented L0 index only when it is absent."""
    _require_confirmation(args, "confirm_write")
    settings = _settings()
    index_name = settings.ES_INDEX_TOPICAL_L0
    configured_dims = settings.EMBEDDING_CONFIGS[settings.EMBEDDING_PROVIDER]["dimensions"]
    if configured_dims != 3072:
        raise SystemExit(
            f"Refusing official OpenAI mapping: configured L0 dimensions are {configured_dims}, not 3072."
        )

    es = _elasticsearch_client()
    if es.indices.exists(index=index_name):
        raise SystemExit(
            f"Index '{index_name}' already exists. The smoke runner never replaces an existing index."
        )

    mapping = {
        "mappings": {
            "properties": {
                "memory_id": {"type": "keyword"},
                "metadata": {
                    "properties": {
                        "user_id": {"type": "keyword"},
                        "role_id": {"type": "keyword"},
                        "created_at": {"type": "date"},
                    }
                },
                "core_fact": {"type": "text"},
                "conclusions": {"type": "object", "enabled": False},
                "keywords": {"type": "keyword"},
                "atomic_units": {
                    "type": "nested",
                    "properties": {
                        "unit_id": {"type": "keyword"},
                        "unit_type": {"type": "keyword"},
                        "content": {"type": "text"},
                        "tag": {"type": "keyword"},
                        "content_vector": {"type": "dense_vector", "dims": 3072},
                    },
                },
            }
        }
    }
    response = es.indices.create(index=index_name, **mapping)
    print(f"Created index '{index_name}':", response.get("acknowledged", False))


def _prepare_l0_run(run_id: str) -> tuple[Path, dict]:
    run_dir = _run_directory(run_id)
    if run_dir.exists():
        raise SystemExit(
            f"Smoke run directory already exists: {run_dir}. Choose a new run-id."
        )
    run_dir.mkdir(parents=True)

    user_id = f"rgmem_smoke_{run_id}"
    session_id = f"smoke_session_{run_id}"
    records = []
    for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["user_id"] = user_id
        record["session_id"] = session_id
        records.append(record)

    input_path = run_dir / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "user_id": user_id,
        "role_id": ROLE_ID,
        "session_id": session_id,
        "input_path": str(input_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _manifest_path(run_id).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return input_path, manifest


def stage_l0(args: argparse.Namespace) -> None:
    """Run the official offline L0 implementation with isolated input/output paths."""
    _require_confirmation(args, "confirm_api_and_write")
    settings = _settings()
    es = _elasticsearch_client()
    if not es.indices.exists(index=settings.ES_INDEX_TOPICAL_L0):
        raise SystemExit("L0 index does not exist. Run the explicit init-es stage first.")

    input_path, manifest = _prepare_l0_run(args.run_id)
    namespace = runpy.run_path(
        str(PROJECT_ROOT / "scripts" / "1_produce_l0_memories.py"),
        run_name=f"rgmem_smoke_l0_{args.run_id}",
    )
    namespace["main"].__globals__.update(
        LOCAL_HISTORY_FILE_PATH=str(input_path),
        BATCH_SIZE=settings.TOPIC_ANALYSIS_BATCH_SIZE or 16,
    )
    namespace["main"]()
    print("L0 smoke stage finished for:", manifest["user_id"])
    print("Artifacts:", _run_directory(args.run_id))


def stage_l1(args: argparse.Namespace) -> None:
    """Run the official offline L1 consumer once for a completed smoke L0 run."""
    _require_confirmation(args, "confirm_api_and_write")
    run_dir = _run_directory(args.run_id)
    _load_manifest(args.run_id)
    memory_ids = run_dir / "l0_memory_ids.txt"
    if not memory_ids.is_file() or not memory_ids.read_text(encoding="utf-8").strip():
        raise SystemExit(f"No L0 memory IDs found at {memory_ids}.")
    _claim_stage(args.run_id, "l1")

    namespace = runpy.run_path(
        str(PROJECT_ROOT / "scripts" / "2_consume_and_build_l1.py"),
        run_name=f"rgmem_smoke_l1_{args.run_id}",
    )
    namespace["main"].__globals__.update(DATA_WORKING_DIRECTORY=str(run_dir))
    namespace["main"]()
    print("L1 smoke stage finished. Artifacts:", run_dir)


def stage_l2(args: argparse.Namespace) -> None:
    """Run one bounded L0+L1 retrieval and save the formatted result."""
    _require_confirmation(args, "confirm_api_and_write")
    manifest = _load_manifest(args.run_id)
    _claim_stage(args.run_id, "l2")

    from config import settings
    from l1_memory.configs.base import MemoryConfig
    from l1_memory.graphs.neo4j import MemoryGraph
    from l1_memory.memory.main import Memory
    from l2_retrieval.l1_adapter import L1Adapter
    from l2_retrieval.retriever import L2Retriever

    config = MemoryConfig(
        llm=settings.L1_LLM_CONFIG,
        embedder=settings.L1_EMBEDDER_CONFIG,
        graph_store=settings.L1_GRAPH_STORE_CONFIG,
    )
    graph = MemoryGraph(config=config)
    retriever = L2Retriever(
        l1_adapter=L1Adapter(l1_memory=Memory(memory_graph=graph))
    )
    entity_queries = [
        {
            "name": "science_fiction_movies",
            "retrieve": ["summary", "inference", "conclusions", "history"],
        },
        {"name": "interstellar", "retrieve": ["conclusions", "history"]},
        {"name": "blade_runner", "retrieve": ["conclusions", "history"]},
    ]
    result = asyncio.run(
        retriever.retrieve(
            query="What kinds of movies does the user prefer and rewatch?",
            user_id=manifest["user_id"],
            role_id=manifest["role_id"],
            retrieval_type="hybrid",
            entity_queries=entity_queries,
        )
    )
    output_path = _run_directory(args.run_id) / "l2_result.txt"
    output_path.write_text(result + "\n", encoding="utf-8")
    print(result)
    print("Saved L2 result to:", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated, explicitly gated RGMem smoke-test runner."
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    preflight = subparsers.add_parser("preflight", help="Read-only service checks.")
    preflight.set_defaults(handler=stage_preflight)

    init_es = subparsers.add_parser(
        "init-es", help="Create the documented L0 index only if absent."
    )
    init_es.add_argument("--confirm-write", action="store_true")
    init_es.set_defaults(handler=stage_init_es)

    for stage_name, handler in (("l0", stage_l0), ("l1", stage_l1), ("l2", stage_l2)):
        stage_parser = subparsers.add_parser(stage_name)
        stage_parser.add_argument("--run-id", required=True)
        stage_parser.add_argument("--confirm-api-and-write", action="store_true")
        stage_parser.set_defaults(handler=handler)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

