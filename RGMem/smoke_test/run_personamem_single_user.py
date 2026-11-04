"""Isolated single-user, multi-round PersonaMem smoke test for RGMem.

The runner reads the original PersonaMem CSV/JSONL files at runtime. It never
modifies RGMem core modules or the source dataset. API/database stages require
an explicit confirmation flag. Each round records L0 additions and L1 graph
ADD/UPDATE diffs in an append-only JSONL audit file.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


SMOKE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SMOKE_ROOT.parent
DATA_ROOT = PROJECT_ROOT.parent / "data"
DEFAULT_QUESTIONS = DATA_ROOT / "questions_32k.csv"
DEFAULT_CONTEXTS = DATA_ROOT / "shared_contexts_32k.jsonl"
ARTIFACTS_ROOT = SMOKE_ROOT / "artifacts"

PERSONA_ID = 0
PERSONA_NAME = "Kanoa Manu"
CONTEXT_ID = "e898d03fec683b1cabf29f57287ff66f8a31842543ecef44b56766844c1c1301"
ROLE_ID = 92001
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
BASE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)
VIRTUAL_FUTURE_CREATED_AT = "2100-01-01T00:00:00.000000+00:00"
PROTOCOL_VERSION = 4

# These are source-message windows, not inferred real-world sessions.
ROUNDS = [
    {"number": 1, "start": 0, "end": 46, "label": "early music preferences and negative shifts"},
    {"number": 2, "start": 47, "end": 77, "label": "renewed theory, podcast, review, and production interests"},
    {"number": 3, "start": 78, "end": 108, "label": "reinforcement and advanced-course withdrawal"},
    {"number": 4, "start": 109, "end": 153, "label": "dating and social preferences"},
    {"number": 5, "start": 154, "end": 182, "label": "collaboration conflict and final music-priority updates"},
]

PROBE_QUERY = (
    "How have the user's preferences about music theory, podcasting, album reviews, "
    "audio production, collaboration, remixing, and original composition changed over time?"
)


def _diagnostic_label_template() -> dict[str, Any]:
    """Candidate A/K labels kept outside every RGMem prompt and write path."""
    return {
        "status": "candidate_requires_human_review",
        "gold_labels_active": False,
        "instructions": [
            "Review each candidate event before using it to judge RGMem.",
            "A contains only preferences that the cited evidence should revise.",
            "K contains preferences that should remain valid outside that scope.",
            "This file is never passed to L0, L1, or L2 generation prompts.",
        ],
        "events": [
            {
                "event_id": "theory-cultural-reengagement",
                "observation_round": 2,
                "s_minus_indices": [22, 24],
                "revision_evidence_indices": [51, 52],
                "candidate_A": [
                    "Music theory becomes appealing when grounded in Pacific music and composition practice."
                ],
                "candidate_K": [
                    "Interest in original composition and Pacific/electronic fusion remains valid.",
                    "Preferences about podcasting, reviews, collaboration, and remixing are not revised by this event.",
                ],
                "probe_A": "Does Kanoa currently want to study music theory, and under what conditions?",
                "probe_K": "What music-creation and cultural-fusion interests remain unchanged?",
                "notes": "Index 51 is an assistant-role record containing embedded 'User:' text; preserve this source anomaly in interpretation.",
            },
            {
                "event_id": "advanced-audio-course-withdrawal",
                "observation_round": 3,
                "s_minus_indices": [70, 71, 92, 93],
                "revision_evidence_indices": [105, 107],
                "candidate_A": [
                    "Kanoa withdraws from the advanced audio-production course because its complexity is overwhelming."
                ],
                "candidate_K": [
                    "This does not by itself erase all interest in practical audio tools or music creation.",
                    "Music theory, podcasting, reviews, collaboration, and cultural fusion remain outside this revision scope.",
                ],
                "probe_A": "Does Kanoa want to continue the advanced audio-production course?",
                "probe_K": "Which practical production and broader music interests should still be retained?",
            },
            {
                "event_id": "band-collaboration-conflict",
                "observation_round": 5,
                "s_minus_indices": [83, 84],
                "revision_evidence_indices": [159],
                "candidate_A": [
                    "The current band collaboration became discouraging because of artistic-direction conflict."
                ],
                "candidate_K": [
                    "The evidence does not justify a global dislike of all collaboration.",
                    "Cross-cultural workshops and collaboration with other musicians can remain positive.",
                ],
                "probe_A": "How does Kanoa feel about continuing the current band collaboration?",
                "probe_K": "Does Kanoa still value other structured or cross-cultural collaborations?",
            },
            {
                "event_id": "remix-hiatus-original-priority",
                "observation_round": 5,
                "s_minus_indices": [155, 157],
                "revision_evidence_indices": [173, 174],
                "candidate_A": [
                    "Kanoa pauses remixing so original compositions and personal projects receive priority."
                ],
                "candidate_K": [
                    "His broader interest in cultural fusion remains valid.",
                    "The hiatus is not evidence that he never enjoyed remixing or will never return to it.",
                    "Music theory and cross-cultural learning remain outside this revision scope.",
                ],
                "probe_A": "What is Kanoa's current priority between remixing and original composition?",
                "probe_K": "Which cultural-fusion and learning interests remain valid during the remix hiatus?",
            },
        ],
    }

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _settings():
    from config import settings

    return settings


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SystemExit(
            "run-id must contain only letters, digits, underscores, or hyphens "
            "and be at most 64 characters."
        )
    return run_id


def _run_directory(run_id: str) -> Path:
    return ARTIFACTS_ROOT / f"personamem-{_validate_run_id(run_id)}"


def _manifest_path(run_id: str) -> Path:
    return _run_directory(run_id) / "manifest.json"


def _load_manifest(run_id: str) -> dict[str, Any]:
    path = _manifest_path(run_id)
    if not path.is_file():
        raise SystemExit(f"Missing manifest: {path}. Run prepare first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_api_and_write:
        raise SystemExit(
            "Refusing API/database work without --confirm-api-and-write. "
            "Run inspect or preflight first."
        )


def _round_spec(round_number: int) -> dict[str, Any]:
    for spec in ROUNDS:
        if spec["number"] == round_number:
            return spec
    raise SystemExit(f"Unknown round {round_number}; expected 1-{len(ROUNDS)}.")


def _round_directory(run_id: str, round_number: int) -> Path:
    return _run_directory(run_id) / f"round-{round_number:02d}"


def _claim_round(run_id: str, round_number: int) -> Path:
    round_dir = _round_directory(run_id, round_number)
    started = round_dir / ".started"
    if started.exists():
        raise SystemExit(
            f"Round {round_number} has already started for {run_id}. "
            "Use a new run-id rather than replaying writes."
        )
    if round_number > 1:
        prior = _round_directory(run_id, round_number - 1) / ".completed"
        if not prior.exists():
            raise SystemExit(f"Round {round_number - 1} is not complete: {prior}")
    started.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return round_dir


def _validate_round_ready(run_id: str, round_number: int) -> Path:
    """Validate local sequencing without claiming or mutating the round."""
    manifest = _load_manifest(run_id)
    round_dir = _round_directory(run_id, round_number)
    if (round_dir / ".started").exists():
        raise SystemExit(
            f"Round {round_number} has already started for {run_id}. "
            "Use a new run-id rather than replaying writes."
        )
    if round_number > 1:
        prior = _round_directory(run_id, round_number - 1) / ".completed"
        if not prior.exists():
            raise SystemExit(f"Round {round_number - 1} is not complete: {prior}")
        if manifest.get("protocol_version", 1) >= 2:
            prior_consolidation = (
                _round_directory(run_id, round_number - 1)
                / "consolidation"
                / ".completed"
            )
            if not prior_consolidation.exists():
                raise SystemExit(
                    f"Round {round_number - 1} consolidation is not complete: "
                    f"{prior_consolidation}"
                )
    input_path = round_dir / "input.jsonl"
    if not input_path.is_file() or not input_path.read_text(encoding="utf-8").strip():
        raise SystemExit(f"Missing or empty round input: {input_path}")
    return round_dir


def _read_contexts(path: Path) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            for context_id, messages in row.items():
                if context_id in contexts:
                    raise ValueError(f"Duplicate shared_context_id: {context_id}")
                if not isinstance(messages, list):
                    raise ValueError(f"Context {context_id} is not a message list")
                contexts[context_id] = messages
    return contexts


def _read_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _source_bundle(questions_path: Path, contexts_path: Path) -> dict[str, Any]:
    contexts = _read_contexts(contexts_path)
    questions = _read_questions(questions_path)
    if CONTEXT_ID not in contexts:
        raise ValueError(f"Selected context is missing: {CONTEXT_ID}")
    messages = contexts[CONTEXT_ID]
    selected_questions = [
        row
        for row in questions
        if int(row["persona_id"]) == PERSONA_ID and row["shared_context_id"] == CONTEXT_ID
    ]
    if len(messages) != 183:
        raise ValueError(f"Expected 183 messages, found {len(messages)}")
    if len(selected_questions) != 9:
        raise ValueError(f"Expected 9 questions, found {len(selected_questions)}")
    if any(int(row["end_index_in_shared_context"]) >= len(messages) for row in selected_questions):
        raise ValueError("A selected question has an out-of-range history boundary")

    system_indices = [i for i, message in enumerate(messages) if message.get("role") == "system"]
    unexpected_roles = [
        (i, message.get("role"))
        for i, message in enumerate(messages)
        if message.get("role") not in {"system", "user", "assistant"}
    ]
    embedded_role_warnings = []
    for i, message in enumerate(messages):
        content = str(message.get("content", ""))
        role = message.get("role")
        if role == "assistant" and re.search(r"(?<!^)User:\s", content):
            embedded_role_warnings.append(
                {"index": i, "role": role, "warning": "assistant_contains_embedded_User"}
            )
        if role == "user" and re.search(r"(?<!^)Assistant:\s", content):
            embedded_role_warnings.append(
                {"index": i, "role": role, "warning": "user_contains_embedded_Assistant"}
            )
    return {
        "messages": messages,
        "questions": selected_questions,
        "system_indices": system_indices,
        "unexpected_roles": unexpected_roles,
        "embedded_role_warnings": embedded_role_warnings,
    }


def _adapt_message(message: dict[str, Any], original_index: int, manifest: dict[str, Any]) -> dict[str, Any]:
    timestamp = BASE_TIMESTAMP + timedelta(minutes=original_index)
    record = {
        "user_id": manifest["user_id"],
        "role_id": manifest["role_id"],
        "role": message.get("role"),
        "content": message.get("content", ""),
        "created_at": timestamp.isoformat(),
        "session_id": manifest["session_id"],
        "original_index": original_index,
        "shared_context_id": CONTEXT_ID,
    }
    if record["role"] == "user":
        record["user_name"] = PERSONA_NAME
    elif record["role"] == "assistant":
        record["assistant_name"] = "PersonaMem Assistant"
    return record


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _prepare_artifacts(run_id: str, questions_path: Path, contexts_path: Path) -> dict[str, Any]:
    run_dir = _run_directory(run_id)
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    bundle = _source_bundle(questions_path, contexts_path)
    if bundle["unexpected_roles"]:
        raise SystemExit(f"Unexpected roles: {bundle['unexpected_roles']}")
    run_dir.mkdir(parents=True)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "study_type": "single-user naturalistic RGMem update-mechanism case",
        "method_variant": "RGMem with strict entity-path schema retry",
        "engineering_compatibility": (
            "entity_paths enforces documented {'path': [...]} objects and retries "
            "a malformed tool call up to two times before invalidating the run"
        ),
        "run_id": run_id,
        "user_id": f"rgmem_personamem_{run_id}",
        "role_id": ROLE_ID,
        "persona_id": PERSONA_ID,
        "persona_name": PERSONA_NAME,
        "shared_context_id": CONTEXT_ID,
        "session_id": f"personamem_context_{CONTEXT_ID}",
        "questions_path": str(questions_path.resolve()),
        "contexts_path": str(contexts_path.resolve()),
        "message_count": len(bundle["messages"]),
        "question_count": len(bundle["questions"]),
        "system_indices_excluded_from_l0": bundle["system_indices"],
        "embedded_role_warnings_preserved": bundle["embedded_role_warnings"],
        "timestamp_policy": "2026-01-01T00:00:00Z plus original_index minutes; synthetic ordering metadata",
        "rounds": ROUNDS,
        "consolidation_schedule": "one scoped native dirty-summary scan after every completed round",
        "diagnostic_labels": "diagnostic_labels.json (candidate only until human review)",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(_manifest_path(run_id), manifest)
    _write_json(run_dir / "questions.json", bundle["questions"])
    _write_json(run_dir / "diagnostic_labels.json", _diagnostic_label_template())
    _write_json(
        run_dir / "source_diagnostics.json",
        {
            "system_indices": bundle["system_indices"],
            "unexpected_roles": bundle["unexpected_roles"],
            "embedded_role_warnings": bundle["embedded_role_warnings"],
        },
    )

    for spec in ROUNDS:
        round_dir = _round_directory(run_id, spec["number"])
        round_dir.mkdir()
        records = []
        excluded = []
        for index in range(spec["start"], spec["end"] + 1):
            message = bundle["messages"][index]
            if message.get("role") == "system":
                excluded.append(index)
                continue
            records.append(_adapt_message(message, index, manifest))
        input_path = round_dir / "input.jsonl"
        input_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        _write_json(
            round_dir / "input_manifest.json",
            {
                **spec,
                "input_path": str(input_path),
                "input_records": len(records),
                "excluded_system_indices": excluded,
                "original_indices": [record["original_index"] for record in records],
            },
        )
    return manifest


def stage_inspect(args: argparse.Namespace) -> None:
    bundle = _source_bundle(args.questions, args.contexts)
    print("Persona:", PERSONA_ID, PERSONA_NAME)
    print("Context:", CONTEXT_ID)
    print("Messages:", len(bundle["messages"]))
    print("Questions:", len(bundle["questions"]))
    print("System indices excluded from L0:", bundle["system_indices"])
    print("Embedded role warnings:", bundle["embedded_role_warnings"])
    for spec in ROUNDS:
        count = sum(
            1
            for index in range(spec["start"], spec["end"] + 1)
            if bundle["messages"][index].get("role") in {"user", "assistant"}
        )
        print(f"Round {spec['number']}: {spec['start']}-{spec['end']} ({count} L0 input records)")


def stage_prepare(args: argparse.Namespace) -> None:
    manifest = _prepare_artifacts(args.run_id, args.questions, args.contexts)
    print("Prepared isolated PersonaMem smoke run:", _run_directory(args.run_id))
    print("Test user:", manifest["user_id"])
    print("Candidate A/K labels (review before judging results):", _run_directory(args.run_id) / "diagnostic_labels.json")
    print("No API or database writes were performed.")


def _elasticsearch_client():
    from elasticsearch import Elasticsearch

    settings = _settings()
    return Elasticsearch(
        settings.ES_CONFIG["host"],
        basic_auth=(settings.ES_CONFIG["user"], settings.ES_CONFIG["password"]),
    )


def stage_preflight(_: argparse.Namespace) -> None:
    import redis
    from neo4j import GraphDatabase

    settings = _settings()
    results = []

    es = _elasticsearch_client()
    try:
        es_ok = bool(es.ping())
        results.append(("Elasticsearch", es_ok, settings.ES_CONFIG["host"]))
        if es_ok:
            print("L0 index:", settings.ES_INDEX_TOPICAL_L0)
            print("L0 index exists:", es.indices.exists(index=settings.ES_INDEX_TOPICAL_L0))
    except Exception as exc:
        results.append(("Elasticsearch", False, f"{type(exc).__name__}: {exc}"))

    redis_client = redis.Redis(**settings.REDIS_CONFIG)
    try:
        results.append(("Redis", bool(redis_client.ping()), "configured endpoint"))
    except Exception as exc:
        results.append(("Redis", False, f"{type(exc).__name__}: {exc}"))

    graph_config = settings.L1_GRAPH_STORE_CONFIG["config"]
    driver = GraphDatabase.driver(
        graph_config["url"],
        auth=(graph_config["username"], graph_config["password"]),
    )
    try:
        try:
            driver.verify_connectivity()
            results.append(("Neo4j", True, graph_config["url"]))
        except Exception as exc:
            results.append(("Neo4j", False, f"{graph_config['url']} ({type(exc).__name__})"))
    finally:
        driver.close()

    for name, ok, detail in results:
        print(f"{name} reachable: {ok} [{detail}]")
    print("L0 LLM provider:", settings.LLM_PROVIDER)
    print("L0 embedding provider:", settings.EMBEDDING_PROVIDER)
    print("L1 LLM provider:", settings.L1_LLM_CONFIG["provider"])
    print("L1 embedding provider:", settings.L1_EMBEDDER_CONFIG["provider"])
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        raise SystemExit("Preflight failed: " + ", ".join(failed))


def _memory_graph():
    from config import settings
    from l1_memory.configs.base import MemoryConfig
    from l1_memory.graphs.neo4j import MemoryGraph

    config = MemoryConfig(
        llm=settings.L1_LLM_CONFIG,
        embedder=settings.L1_EMBEDDER_CONFIG,
        graph_store=settings.L1_GRAPH_STORE_CONFIG,
    )
    return MemoryGraph(config=config)


def _graph_snapshot(graph, user_id: str, role_id: int) -> dict[str, Any]:
    node_rows = graph.graph.query(
        """
        MATCH (n {user_id: $user_id, role_id: $role_id})
        RETURN elementId(n) AS id, n.name AS name, labels(n) AS labels,
               n.type AS type, n.created AS created,
               coalesce(n.history, []) AS history,
               size(coalesce(n.history, [])) AS history_count,
               n.mentions AS mentions, n.community_influence_score AS community_influence_score,
               n.is_dirty AS is_dirty, n.summary AS summary,
               n.last_summary_update AS last_summary_update
        """,
        params={"user_id": user_id, "role_id": role_id},
    )
    relationship_rows = graph.graph.query(
        """
        MATCH (source {user_id: $user_id, role_id: $role_id})
              -[r:RELATED_TO]->
              (destination {user_id: $user_id, role_id: $role_id})
        RETURN elementId(r) AS id, source.name AS source, destination.name AS destination,
               r.type AS type, r.created AS created,
               coalesce(r.history, []) AS history,
               coalesce(r.l1_conclusions, []) AS l1_conclusions,
               size(coalesce(r.history, [])) AS history_count,
               size(coalesce(r.l1_conclusions, [])) AS conclusion_count,
               r.mentions AS mentions, r.inference AS inference,
               r.last_inference_update AS last_inference_update
        """,
        params={"user_id": user_id, "role_id": role_id},
    )
    return {
        "nodes": {row["id"]: row for row in node_rows},
        "relationships": {row["id"]: row for row in relationship_rows},
    }


def _changed_fields(before: dict[str, Any], after: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    changes = {}
    for field in fields:
        if before.get(field) != after.get(field):
            changes[field] = {"before": before.get(field), "after": after.get(field)}
    return changes


def _snapshot_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    events: dict[str, list[dict[str, Any]]] = {
        "add_nodes": [],
        "update_nodes": [],
        "add_relationships": [],
        "update_relationships": [],
    }
    for node_id, node in after["nodes"].items():
        if node_id not in before["nodes"]:
            events["add_nodes"].append(node)
            continue
        changes = _changed_fields(
            before["nodes"][node_id],
            node,
            [
                "type",
                "history",
                "history_count",
                "mentions",
                "community_influence_score",
                "is_dirty",
                "summary",
                "last_summary_update",
            ],
        )
        if changes:
            events["update_nodes"].append(
                {"id": node_id, "name": node.get("name"), "changes": changes}
            )
    for relationship_id, relationship in after["relationships"].items():
        if relationship_id not in before["relationships"]:
            events["add_relationships"].append(relationship)
            continue
        changes = _changed_fields(
            before["relationships"][relationship_id],
            relationship,
            [
                "type",
                "history",
                "l1_conclusions",
                "history_count",
                "conclusion_count",
                "mentions",
                "inference",
                "last_inference_update",
            ],
        )
        if changes:
            events["update_relationships"].append(
                {
                    "id": relationship_id,
                    "source": relationship.get("source"),
                    "destination": relationship.get("destination"),
                    "changes": changes,
                }
            )
    return events


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _l1_write_outcome(
    graph_add_result: Any,
    diff: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Classify a graph.add call without changing RGMem's write behavior."""
    write_results = (
        graph_add_result.get("write_results")
        if isinstance(graph_add_result, dict)
        else None
    )
    added_entities = (
        write_results.get("added_entities")
        if isinstance(write_results, dict)
        else None
    )
    mutation_counts = {
        key: len(diff.get(key, []))
        for key in (
            "add_nodes",
            "update_nodes",
            "add_relationships",
            "update_relationships",
        )
    }
    graph_mutation_count = sum(mutation_counts.values())
    returned_entity_path_count = len(added_entities) if isinstance(added_entities, list) else None
    empty_write = graph_mutation_count == 0 and not added_entities
    return {
        "status": "empty_write" if empty_write else "accepted",
        "accepted": not empty_write,
        "graph_mutation_count": graph_mutation_count,
        "mutation_counts": mutation_counts,
        "returned_entity_path_count": returned_entity_path_count,
        "reason": (
            "graph.add returned no entity paths and produced no observable graph mutation"
            if empty_write
            else "graph.add returned entity paths or produced an observable graph mutation"
        ),
    }


def _input_timestamp_index(input_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index[record["created_at"]] = {
            "original_index": record.get("original_index"),
            "role": record.get("role"),
        }
    return index


def _turn_trace(turn: dict[str, Any], timestamp_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    created_at = turn.get("created_at")
    source = timestamp_index.get(created_at, {})
    return {
        "created_at": created_at,
        "original_index": source.get("original_index"),
        "role": turn.get("role") or source.get("role"),
        "virtual_future_turn": created_at == VIRTUAL_FUTURE_CREATED_AT,
    }


def _run_l0(input_path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Run the official L0 script while recording its exact segmentation boundaries."""
    settings = _settings()
    timestamp_index = _input_timestamp_index(input_path)
    trace: dict[str, Any] = {
        "trace_version": 1,
        "input_path": str(input_path),
        "virtual_future_created_at": VIRTUAL_FUTURE_CREATED_AT,
        "split_calls": [],
        "summary_calls": [],
    }

    from l0_memory.memory_generation import processor as l0_processor_functions

    original_split: Callable[..., list] = l0_processor_functions.split_topics_from_chunk
    original_summary: Callable[..., dict] = l0_processor_functions.generate_l0_summary

    def traced_split(dialogue_chunk, llm_client, model_override=None):
        topics = original_split(dialogue_chunk, llm_client, model_override=model_override)
        trace["split_calls"].append(
            {
                "call": len(trace["split_calls"]) + 1,
                "input_turns": [
                    _turn_trace(turn, timestamp_index) for turn in dialogue_chunk
                ],
                "identified_topics": _json_safe(topics),
            }
        )
        return topics

    def traced_summary(
        topic_dialogue,
        user_id,
        role_id,
        start_created_at,
        llm_client,
        model_override=None,
    ):
        result = original_summary(
            topic_dialogue=topic_dialogue,
            user_id=user_id,
            role_id=role_id,
            start_created_at=start_created_at,
            llm_client=llm_client,
            model_override=model_override,
        )
        turns = [_turn_trace(turn, timestamp_index) for turn in topic_dialogue]
        source_indices = [
            turn["original_index"]
            for turn in turns
            if turn["original_index"] is not None
        ]
        user_indices = [
            turn["original_index"]
            for turn in turns
            if turn["original_index"] is not None and turn["role"] == "user"
        ]
        context_indices = [
            turn["original_index"]
            for turn in turns
            if turn["original_index"] is not None and turn["role"] == "assistant"
        ]
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        trace["summary_calls"].append(
            {
                "call": len(trace["summary_calls"]) + 1,
                "memory_id": result.get("memory_id") if isinstance(result, dict) else None,
                "topic_start_created_at": start_created_at,
                "topic_end_created_at": turns[-1]["created_at"] if turns else None,
                "topic_turns": turns,
                "source_indices": source_indices,
                "source_index_range": (
                    [min(source_indices), max(source_indices)] if source_indices else None
                ),
                "user_evidence_indices": user_indices,
                "assistant_context_indices": context_indices,
                "contains_virtual_future_turn": any(
                    turn["virtual_future_turn"] for turn in turns
                ),
                "llm_metadata_created_at": metadata.get("created_at"),
                "llm_metadata_created_at_range": metadata.get("created_at_range"),
            }
        )
        return result

    l0_processor_functions.split_topics_from_chunk = traced_split
    l0_processor_functions.generate_l0_summary = traced_summary
    try:
        namespace = runpy.run_path(
            str(PROJECT_ROOT / "scripts" / "1_produce_l0_memories.py"),
            run_name=f"rgmem_personamem_l0_{input_path.parent.name}",
        )
        namespace["main"].__globals__.update(
            LOCAL_HISTORY_FILE_PATH=str(input_path),
            BATCH_SIZE=settings.TOPIC_ANALYSIS_BATCH_SIZE or 16,
        )
        namespace["main"]()
    finally:
        l0_processor_functions.split_topics_from_chunk = original_split
        l0_processor_functions.generate_l0_summary = original_summary
        _write_json(input_path.parent / "l0_provenance_trace.json", trace)

    memory_ids_path = input_path.parent / "l0_memory_ids.txt"
    if not memory_ids_path.is_file():
        raise RuntimeError(f"L0 did not create {memory_ids_path}")
    memory_ids = [line.strip() for line in memory_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not memory_ids:
        raise RuntimeError(f"L0 produced no memory IDs for {input_path}")
    provenance_by_memory_id = {
        item["memory_id"]: item
        for item in trace["summary_calls"]
        if item.get("memory_id")
    }
    return memory_ids, provenance_by_memory_id


def _assert_new_user_is_empty(manifest: dict[str, Any], graph) -> None:
    settings = _settings()
    es = _elasticsearch_client()
    if es.indices.exists(index=settings.ES_INDEX_TOPICAL_L0):
        result = es.count(
            index=settings.ES_INDEX_TOPICAL_L0,
            query={"term": {"metadata.user_id": manifest["user_id"]}},
        )
        if result.get("count", 0):
            raise RuntimeError(
                f"Elasticsearch already contains {result['count']} documents for {manifest['user_id']}"
            )
    snapshot = _graph_snapshot(graph, manifest["user_id"], manifest["role_id"])
    if snapshot["nodes"] or snapshot["relationships"]:
        raise RuntimeError(f"Neo4j already contains data for {manifest['user_id']}")


def _require_l0_index() -> None:
    settings = _settings()
    es = _elasticsearch_client()
    if not es.ping():
        raise SystemExit("Elasticsearch is not reachable; no round was claimed.")
    if not es.indices.exists(index=settings.ES_INDEX_TOPICAL_L0):
        raise SystemExit(
            f"L0 index '{settings.ES_INDEX_TOPICAL_L0}' does not exist; "
            "initialize it with the existing smoke runner before starting round 1."
        )


def _top_entity_queries(graph, user_id: str, role_id: int) -> list[dict[str, Any]]:
    rows = graph.graph.query(
        """
        MATCH (n:abstract {user_id: $user_id, role_id: $role_id})
        OPTIONAL MATCH (n)-[r:RELATED_TO]->()
        RETURN n.name AS name, count(r) AS degree
        ORDER BY degree DESC, name ASC
        LIMIT 6
        """,
        params={"user_id": user_id, "role_id": role_id},
    )
    return [
        {"name": row["name"], "retrieve": ["summary", "inference", "conclusions", "history"]}
        for row in rows
        if row.get("name")
    ]


def _diagnostic_probes(run_dir: Path, round_number: int) -> list[dict[str, str]]:
    labels_path = run_dir / "diagnostic_labels.json"
    if not labels_path.is_file():
        return []
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    probes = []
    for event in labels.get("events", []):
        if int(event.get("observation_round", 0)) > round_number:
            continue
        for scope, key in (("A", "probe_A"), ("K", "probe_K")):
            query = event.get(key)
            if query:
                probes.append(
                    {
                        "probe_id": f"{event['event_id']}:{scope}",
                        "event_id": event["event_id"],
                        "scope": scope,
                        "query": query,
                    }
                )
    return probes


def _run_l2_probe(
    graph,
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    round_number: int,
    phase: str,
    include_diagnostic_probes: bool,
) -> str:
    from l1_memory.memory.main import Memory
    from l2_retrieval.l1_adapter import L1Adapter
    from l2_retrieval.retriever import L2Retriever

    entity_queries = _top_entity_queries(graph, manifest["user_id"], manifest["role_id"])
    _write_json(output_dir / "l2_entity_queries.json", entity_queries)
    retriever = L2Retriever(l1_adapter=L1Adapter(l1_memory=Memory(memory_graph=graph)))
    result = asyncio.run(
        retriever.retrieve(
            query=PROBE_QUERY,
            user_id=manifest["user_id"],
            role_id=manifest["role_id"],
            retrieval_type="hybrid",
            entity_queries=entity_queries or None,
            keywords=None if entity_queries else "music theory podcast album review collaboration remix original composition",
        )
    )
    (output_dir / "l2_result.txt").write_text(result + "\n", encoding="utf-8")

    diagnostic_results = []
    probes = (
        _diagnostic_probes(_run_directory(manifest["run_id"]), round_number)
        if include_diagnostic_probes
        else []
    )
    for probe in probes:
        probe_result = asyncio.run(
            retriever.retrieve(
                query=probe["query"],
                user_id=manifest["user_id"],
                role_id=manifest["role_id"],
                retrieval_type="relevance",
                entity_queries=entity_queries or None,
                keywords=(
                    None
                    if entity_queries
                    else "music theory podcast album review collaboration remix original composition"
                ),
            )
        )
        diagnostic_results.append({**probe, "result": probe_result})
    labels_path = _run_directory(manifest["run_id"]) / "diagnostic_labels.json"
    label_status = (
        json.loads(labels_path.read_text(encoding="utf-8")).get("status", "unknown")
        if labels_path.is_file()
        else "not_available_for_legacy_run"
    )
    _write_json(
        output_dir / "l2_diagnostic_probes.json",
        {
            "round": round_number,
            "phase": phase,
            "label_status": label_status,
            "enabled": include_diagnostic_probes,
            "results": diagnostic_results,
        },
    )
    return result


def _raw_l0_source(loader, memory_id: str) -> dict[str, Any]:
    response = loader.es_client.get(index=loader.index_name, id=memory_id)
    source = response.get("_source") or {}
    if not source:
        raise RuntimeError(f"Elasticsearch returned an empty source for {memory_id}")
    return source


def _write_checkpoint(
    checkpoints_dir: Path,
    filename: str,
    *,
    snapshot: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    path = checkpoints_dir / filename
    _write_json(path, {"metadata": metadata, "snapshot": snapshot})
    return path


def _graph_add_with_trigger_trace(
    graph,
    *,
    input_data: dict[str, Any],
    filters: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    """Observe threshold checks without changing RGMem's implementation."""
    checks: list[dict[str, Any]] = []
    original_inference_check = graph.check_and_trigger_inference
    original_summary_check = graph.check_and_trigger_summary

    def traced_inference_check(relationship_id, new_mentions, event_timestamp):
        checks.append(
            {
                "kind": "inference_threshold_check",
                "target_id": relationship_id,
                "observed_value": new_mentions,
                "threshold": graph.inference_threshold,
                "threshold_reached": new_mentions >= graph.inference_threshold,
                "event_timestamp": event_timestamp,
            }
        )
        return original_inference_check(relationship_id, new_mentions, event_timestamp)

    def traced_summary_check(abstract_node_id, new_community_score, event_timestamp):
        checks.append(
            {
                "kind": "community_summary_threshold_check",
                "target_id": abstract_node_id,
                "observed_value": new_community_score,
                "threshold": graph.summary_threshold,
                "threshold_reached": new_community_score >= graph.summary_threshold,
                "event_timestamp": event_timestamp,
            }
        )
        return original_summary_check(
            abstract_node_id,
            new_community_score,
            event_timestamp,
        )

    graph.check_and_trigger_inference = traced_inference_check
    graph.check_and_trigger_summary = traced_summary_check
    try:
        result = graph.add(input_data=input_data, filters=filters)
    finally:
        graph.check_and_trigger_inference = original_inference_check
        graph.check_and_trigger_summary = original_summary_check
    return result, checks


def stage_round(args: argparse.Namespace) -> None:
    _require_confirmation(args)
    manifest = _load_manifest(args.run_id)
    spec = _round_spec(args.round)
    round_dir = _validate_round_ready(args.run_id, args.round)
    input_path = round_dir / "input.jsonl"
    _require_l0_index()
    graph = _memory_graph()
    if args.round == 1:
        _assert_new_user_is_empty(manifest, graph)

    # Claim only after all non-mutating local checks and service/index checks pass.
    round_dir = _claim_round(args.run_id, args.round)

    before_round = _graph_snapshot(graph, manifest["user_id"], manifest["role_id"])
    checkpoints_dir = round_dir / "checkpoints"
    checkpoints_dir.mkdir()
    _write_checkpoint(
        checkpoints_dir,
        "round-before.json",
        snapshot=before_round,
        metadata={"round": args.round, "phase": "before_l0_to_l1"},
    )
    memory_ids, provenance_by_memory_id = _run_l0(input_path)
    audit_path = round_dir / "add_update_audit.jsonl"
    from l1_memory.data_loader import L0MemoryLoader

    loader = L0MemoryLoader()
    current = before_round
    for sequence, memory_id in enumerate(memory_ids, 1):
        memory_data = loader.get_memory_by_id(memory_id)
        if not memory_data:
            raise RuntimeError(f"Could not load L0 memory {memory_id}")
        if memory_data.get("user_id") != manifest["user_id"]:
            raise RuntimeError(f"Unexpected user_id in {memory_id}: {memory_data.get('user_id')}")
        raw_source = _raw_l0_source(loader, memory_id)
        provenance = provenance_by_memory_id.get(memory_id)
        before_checkpoint = _write_checkpoint(
            checkpoints_dir,
            f"l0-{sequence:02d}-before-l1.json",
            snapshot=current,
            metadata={
                "round": args.round,
                "sequence": sequence,
                "phase": "before_graph_add",
                "memory_id": memory_id,
                "l0_provenance": provenance,
            },
        )
        _append_jsonl(
            audit_path,
            {
                "operation": "ADD_L0_MEMORY",
                "round": args.round,
                "sequence": sequence,
                "round_source_window": [spec["start"], spec["end"]],
                "source_index_range": (
                    provenance.get("source_index_range") if provenance else None
                ),
                "user_evidence_indices": (
                    provenance.get("user_evidence_indices") if provenance else None
                ),
                "assistant_context_indices": (
                    provenance.get("assistant_context_indices") if provenance else None
                ),
                "topic_split_boundary": (
                    {
                        "start_created_at": provenance.get("topic_start_created_at"),
                        "end_created_at": provenance.get("topic_end_created_at"),
                    }
                    if provenance
                    else None
                ),
                "llm_metadata_created_at_range": (
                    provenance.get("llm_metadata_created_at_range") if provenance else None
                ),
                "memory_id": memory_id,
                "created_at": memory_data.get("created_at"),
                "raw_metadata": raw_source.get("metadata", {}),
                "core_fact": memory_data.get("core_fact"),
                "conclusions": memory_data.get("conclusions", []),
            },
        )
        graph_add_result, trigger_checks = _graph_add_with_trigger_trace(
            graph,
            input_data=memory_data,
            filters={"user_id": manifest["user_id"], "role_id": manifest["role_id"]},
        )
        after = _graph_snapshot(graph, manifest["user_id"], manifest["role_id"])
        diff = _snapshot_diff(current, after)
        l1_write_outcome = _l1_write_outcome(graph_add_result, diff)
        after_checkpoint = _write_checkpoint(
            checkpoints_dir,
            f"l0-{sequence:02d}-after-l1.json",
            snapshot=after,
            metadata={
                "round": args.round,
                "sequence": sequence,
                "phase": "after_graph_add",
                "memory_id": memory_id,
                "execution_boundary": (
                    "graph.add includes L1 writes plus any threshold-triggered "
                    "inference/community-summary evolution"
                ),
            },
        )
        _append_jsonl(
            audit_path,
            {
                "operation": "L1_ADD_UPDATE",
                "round": args.round,
                "sequence": sequence,
                "memory_id": memory_id,
                "execution_boundary": "joint_graph_add_and_auto_evolution",
                "threshold_checks": trigger_checks,
                "graph_add_result": _json_safe(graph_add_result),
                "l1_write_outcome": l1_write_outcome,
                "checkpoint_before": str(before_checkpoint),
                "checkpoint_after": str(after_checkpoint),
                **diff,
            },
        )
        if not l1_write_outcome["accepted"]:
            failed_at = datetime.now(timezone.utc).isoformat()
            failure = {
                "failure_type": "L1_EMPTY_WRITE",
                "round": args.round,
                "sequence": sequence,
                "memory_id": memory_id,
                "source_index_range": (
                    provenance.get("source_index_range") if provenance else None
                ),
                "l1_write_outcome": l1_write_outcome,
                "graph_add_result": _json_safe(graph_add_result),
                "checkpoint_before": str(before_checkpoint),
                "checkpoint_after": str(after_checkpoint),
                "failed_at": failed_at,
                "next_action": (
                    "Preserve this run as failure evidence. Do not replay writes on the same "
                    "run-id; use a new isolated run-id for another full trajectory."
                ),
            }
            _write_json(round_dir / "round_failure.json", failure)
            (round_dir / ".failed").write_text(failed_at, encoding="utf-8")
            raise RuntimeError(
                f"RGMem L1 empty write in round {args.round}, L0 sequence {sequence}, "
                f"memory {memory_id}. Evidence: {round_dir / 'round_failure.json'}"
            )
        current = after

    l2_result = _run_l2_probe(
        graph,
        manifest,
        round_dir,
        round_number=args.round,
        phase="immediate_after_round_before_dirty_summary",
        include_diagnostic_probes=args.with_diagnostic_probes,
    )
    round_diff = _snapshot_diff(before_round, current)
    _write_checkpoint(
        checkpoints_dir,
        "round-after.json",
        snapshot=current,
        metadata={"round": args.round, "phase": "after_l1_before_dirty_consolidation"},
    )
    summary = {
        "round": args.round,
        "label": spec["label"],
        "source_index_range": [spec["start"], spec["end"]],
        "l0_memory_ids": memory_ids,
        "l0_memory_count": len(memory_ids),
        "l1_write_count": len(memory_ids),
        "l1_empty_write_count": 0,
        "l0_provenance_trace": "l0_provenance_trace.json",
        "checkpoint_directory": "checkpoints",
        "l1_execution_boundary": "joint_graph_add_and_auto_evolution",
        "nodes_before": len(before_round["nodes"]),
        "nodes_after": len(current["nodes"]),
        "relationships_before": len(before_round["relationships"]),
        "relationships_after": len(current["relationships"]),
        "round_add_update": round_diff,
        "l2_result_chars": len(l2_result),
        "diagnostic_probes_enabled": args.with_diagnostic_probes,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(round_dir / "round_summary.json", summary)
    (round_dir / ".completed").write_text(summary["completed_at"], encoding="utf-8")
    print(f"Round {args.round} completed for {manifest['user_id']}")
    print("L0 memories added:", len(memory_ids))
    print("Nodes:", len(before_round["nodes"]), "->", len(current["nodes"]))
    print("Relationships:", len(before_round["relationships"]), "->", len(current["relationships"]))
    print("ADD/UPDATE audit:", audit_path)
    print("L2 result:", round_dir / "l2_result.txt")


def _dirty_abstract_nodes(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in snapshot["nodes"].values()
        if node.get("type") == "abstract" and node.get("is_dirty") is True
    ]


def stage_consolidate(args: argparse.Namespace) -> None:
    """Run one user-scoped native dirty-summary scan and record actual rewrites."""
    _require_confirmation(args)
    manifest = _load_manifest(args.run_id)
    round_dir = _round_directory(args.run_id, args.after_round)
    if not (round_dir / ".completed").exists():
        raise SystemExit(f"Round {args.after_round} is not complete: {round_dir / '.completed'}")
    if args.after_round < len(ROUNDS):
        next_started = _round_directory(args.run_id, args.after_round + 1) / ".started"
        if next_started.exists():
            raise SystemExit(
                "Cannot isolate this consolidation after the next round has started: "
                f"{next_started}"
            )

    consolidation_dir = round_dir / "consolidation"
    if (consolidation_dir / ".started").exists():
        raise SystemExit(
            f"Consolidation after round {args.after_round} has already started. "
            "Preserve the evidence and use a new run-id instead of replaying it."
        )

    graph = _memory_graph()
    before = _graph_snapshot(graph, manifest["user_id"], manifest["role_id"])
    dirty_before = _dirty_abstract_nodes(before)

    consolidation_dir.mkdir()
    started_at = datetime.now(timezone.utc).isoformat()
    (consolidation_dir / ".started").write_text(started_at, encoding="utf-8")
    _write_json(
        consolidation_dir / "before.json",
        {
            "metadata": {
                "track": "rgmem_native_user_scoped_dirty_summary",
                "after_round": args.after_round,
                "started_at": started_at,
            },
            "snapshot": before,
        },
    )

    invoked_processor = bool(dirty_before)
    if invoked_processor:
        graph.summary_processor.summarize_dirty_nodes(
            user_id=manifest["user_id"],
            role_id=manifest["role_id"],
            trigger_timestamp=started_at,
        )

    after = _graph_snapshot(graph, manifest["user_id"], manifest["role_id"])
    dirty_after = _dirty_abstract_nodes(after)
    diff = _snapshot_diff(before, after)
    rewritten_nodes = [
        item
        for item in diff["update_nodes"]
        if "summary" in item.get("changes", {})
    ]
    l2_result = ""
    if invoked_processor:
        l2_result = _run_l2_probe(
            graph,
            manifest,
            consolidation_dir,
            round_number=args.after_round,
            phase="after_dirty_summary",
            include_diagnostic_probes=args.with_diagnostic_probes,
        )
    completed_at = datetime.now(timezone.utc).isoformat()
    report = {
        "track": "rgmem_native_user_scoped_dirty_summary",
        "after_round": args.after_round,
        "processor_invoked": invoked_processor,
        "dirty_nodes_before": [
            {"id": node.get("id"), "name": node.get("name")} for node in dirty_before
        ],
        "dirty_nodes_after": [
            {"id": node.get("id"), "name": node.get("name")} for node in dirty_after
        ],
        "effective_summary_rewrite_count": len(rewritten_nodes),
        "rewritten_nodes": rewritten_nodes,
        "effective_consolidation": bool(rewritten_nodes),
        "l2_probe_executed": invoked_processor,
        "l2_result_chars": len(l2_result),
        "l2_diagnostic_probes": (
            "l2_diagnostic_probes.json" if invoked_processor else None
        ),
        "diagnostic_probes_enabled": args.with_diagnostic_probes,
        "interpretation": (
            "effective native hierarchical rewrites observed"
            if rewritten_nodes
            else "scan completed without a summary rewrite; do not count as consolidation pressure"
        ),
        "snapshot_diff": diff,
        "completed_at": completed_at,
    }
    _write_json(
        consolidation_dir / "after.json",
        {
            "metadata": {
                "track": "rgmem_native_user_scoped_dirty_summary",
                "after_round": args.after_round,
                "completed_at": completed_at,
            },
            "snapshot": after,
        },
    )
    _write_json(consolidation_dir / "consolidation_report.json", report)
    (consolidation_dir / ".completed").write_text(completed_at, encoding="utf-8")

    print(f"Consolidation scan after round {args.after_round} completed.")
    print("Dirty abstract nodes:", len(dirty_before), "->", len(dirty_after))
    print("Actual summary rewrites:", len(rewritten_nodes))
    print("Report:", consolidation_dir / "consolidation_report.json")


def stage_status(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.run_id)
    print("Run:", manifest["run_id"])
    print("User:", manifest["user_id"])
    print("Context:", manifest["shared_context_id"])
    for spec in ROUNDS:
        round_dir = _round_directory(args.run_id, spec["number"])
        if (round_dir / ".completed").exists():
            state = "completed"
        elif (round_dir / ".failed").exists():
            state = "failed"
        elif (round_dir / ".started").exists():
            state = "started/incomplete"
        else:
            state = "not started"
        summary_path = round_dir / "round_summary.json"
        details = ""
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            details = (
                f"; L0={summary['l0_memory_count']}, nodes={summary['nodes_after']}, "
                f"relationships={summary['relationships_after']}"
            )
        failure_path = round_dir / "round_failure.json"
        if failure_path.is_file():
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            details += (
                f"; failure={failure.get('failure_type')}"
                f"@L0-{int(failure.get('sequence', 0)):02d}"
            )
        consolidation_dir = round_dir / "consolidation"
        if (consolidation_dir / ".completed").exists():
            consolidation_state = "completed"
            report_path = consolidation_dir / "consolidation_report.json"
            if report_path.is_file():
                consolidation_report = json.loads(report_path.read_text(encoding="utf-8"))
                consolidation_state += (
                    f"; rewrites={consolidation_report['effective_summary_rewrite_count']}"
                )
        elif (consolidation_dir / ".started").exists():
            consolidation_state = "started/incomplete"
        else:
            consolidation_state = "not started"
        print(
            f"Round {spec['number']}: {state}{details}; "
            f"consolidation={consolidation_state}"
        )


def _compact_add_update_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("operation") == "ADD_L0_MEMORY":
        return {
            "operation": "ADD_L0_MEMORY",
            "round": event.get("round"),
            "sequence": event.get("sequence"),
            "memory_id": event.get("memory_id"),
            "created_at": event.get("created_at"),
            "source_index_range": event.get("source_index_range"),
            "user_evidence_indices": event.get("user_evidence_indices"),
            "assistant_context_indices": event.get("assistant_context_indices"),
            "topic_split_boundary": event.get("topic_split_boundary"),
            "llm_metadata_created_at_range": event.get("llm_metadata_created_at_range"),
            "core_fact": event.get("core_fact"),
        }
    diff = {
        key: event.get(key, [])
        for key in (
            "add_nodes",
            "update_nodes",
            "add_relationships",
            "update_relationships",
        )
    }
    l1_write_outcome = event.get("l1_write_outcome") or _l1_write_outcome(
        event.get("graph_add_result"),
        diff,
    )
    return {
        "operation": "L1_ADD_UPDATE",
        "round": event.get("round"),
        "sequence": event.get("sequence"),
        "memory_id": event.get("memory_id"),
        "execution_boundary": event.get("execution_boundary"),
        "l1_write_outcome": l1_write_outcome,
        "threshold_checks": event.get("threshold_checks", []),
        "add_nodes": [item.get("name") for item in event.get("add_nodes", [])],
        "update_nodes": [item.get("name") for item in event.get("update_nodes", [])],
        "add_relationships": [
            {"source": item.get("source"), "destination": item.get("destination"), "type": item.get("type")}
            for item in event.get("add_relationships", [])
        ],
        "update_relationships": [
            {
                "source": item.get("source"),
                "destination": item.get("destination"),
                "changes": item.get("changes", {}),
            }
            for item in event.get("update_relationships", [])
        ],
    }


def stage_report(args: argparse.Namespace) -> None:
    """Aggregate available ADD/UPDATE logs without API or database access."""
    manifest = _load_manifest(args.run_id)
    run_dir = _run_directory(args.run_id)
    all_events = []
    completed_rounds = []
    failed_rounds = []
    observed_rounds = []
    consolidations = []
    for spec in ROUNDS:
        round_dir = _round_directory(args.run_id, spec["number"])
        audit_path = round_dir / "add_update_audit.jsonl"
        if (round_dir / ".completed").exists():
            completed_rounds.append(spec["number"])
        if (round_dir / ".failed").exists():
            failed_rounds.append(spec["number"])
        if not audit_path.is_file():
            continue
        observed_rounds.append(spec["number"])
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_events.append(_compact_add_update_event(json.loads(line)))
        consolidation_report = round_dir / "consolidation" / "consolidation_report.json"
        if consolidation_report.is_file():
            consolidations.append(
                json.loads(consolidation_report.read_text(encoding="utf-8"))
            )

    operation_counts: dict[str, int] = {}
    add_node_count = update_node_count = 0
    add_relationship_count = update_relationship_count = 0
    empty_l1_write_count = 0
    for event in all_events:
        operation = event["operation"]
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
        if event.get("l1_write_outcome", {}).get("status") == "empty_write":
            empty_l1_write_count += 1
        add_node_count += len(event.get("add_nodes", []))
        update_node_count += len(event.get("update_nodes", []))
        add_relationship_count += len(event.get("add_relationships", []))
        update_relationship_count += len(event.get("update_relationships", []))

    report = {
        "run_id": manifest["run_id"],
        "user_id": manifest["user_id"],
        "shared_context_id": manifest["shared_context_id"],
        "completed_rounds": completed_rounds,
        "failed_rounds": failed_rounds,
        "observed_rounds": observed_rounds,
        "operation_counts": operation_counts,
        "diagnostic_label_status": (
            json.loads(
                (run_dir / "diagnostic_labels.json").read_text(encoding="utf-8")
            ).get("status")
            if (run_dir / "diagnostic_labels.json").is_file()
            else "not_available_for_legacy_run"
        ),
        "consolidations": consolidations,
        "totals": {
            "add_nodes": add_node_count,
            "update_nodes": update_node_count,
            "add_relationships": add_relationship_count,
            "update_relationships": update_relationship_count,
            "empty_l1_writes": empty_l1_write_count,
        },
        "events": all_events,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path = run_dir / "add_update_report.json"
    _write_json(json_path, report)

    markdown = [
        "# PersonaMem single-user ADD/UPDATE report",
        "",
        f"- Run: `{manifest['run_id']}`",
        f"- User: `{manifest['user_id']}`",
        f"- Context: `{manifest['shared_context_id']}`",
        f"- Completed rounds: `{completed_rounds}`",
        f"- Failed rounds: `{failed_rounds}`",
        f"- Rounds with audit evidence: `{observed_rounds}`",
        f"- L0 memory additions: `{operation_counts.get('ADD_L0_MEMORY', 0)}`",
        f"- L1 node ADD operations: `{add_node_count}`",
        f"- L1 node UPDATE operations: `{update_node_count}`",
        f"- L1 relationship ADD operations: `{add_relationship_count}`",
        f"- L1 relationship UPDATE operations: `{update_relationship_count}`",
        f"- L1 empty writes: `{empty_l1_write_count}`",
        f"- Dirty-summary scans recorded: `{len(consolidations)}`",
        f"- Effective summary rewrites: `{sum(item.get('effective_summary_rewrite_count', 0) for item in consolidations)}`",
        "",
        "The full per-memory provenance, changed fields, and consolidation evidence are stored in `add_update_report.json`.",
        "",
    ]
    markdown_path = run_dir / "add_update_report.md"
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print("ADD/UPDATE JSON report:", json_path)
    print("ADD/UPDATE Markdown summary:", markdown_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-user multi-round PersonaMem RGMem smoke test")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Read-only source validation")
    inspect_parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    inspect_parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    inspect_parser.set_defaults(handler=stage_inspect)

    preflight = subparsers.add_parser("preflight", help="Read-only service checks")
    preflight.set_defaults(handler=stage_preflight)

    prepare = subparsers.add_parser("prepare", help="Create isolated round inputs; no API/database writes")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    prepare.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    prepare.set_defaults(handler=stage_prepare)

    round_parser = subparsers.add_parser("round", help="Run one sequential L0 -> L1 -> L2 round")
    round_parser.add_argument("--run-id", required=True)
    round_parser.add_argument("--round", type=int, required=True, choices=range(1, len(ROUNDS) + 1))
    round_parser.add_argument("--confirm-api-and-write", action="store_true")
    round_parser.add_argument(
        "--with-diagnostic-probes",
        action="store_true",
        help="Run the candidate A/K retrieval probes in addition to the one generic L2 probe",
    )
    round_parser.set_defaults(handler=stage_round)

    consolidate = subparsers.add_parser(
        "consolidate",
        help="Run one user-scoped native dirty-summary scan after a completed round",
    )
    consolidate.add_argument("--run-id", required=True)
    consolidate.add_argument(
        "--after-round",
        type=int,
        required=True,
        choices=range(1, len(ROUNDS) + 1),
    )
    consolidate.add_argument("--confirm-api-and-write", action="store_true")
    consolidate.add_argument(
        "--with-diagnostic-probes",
        action="store_true",
        help="Run the candidate A/K retrieval probes after dirty-summary processing",
    )
    consolidate.set_defaults(handler=stage_consolidate)

    status = subparsers.add_parser("status", help="Read local run status")
    status.add_argument("--run-id", required=True)
    status.set_defaults(handler=stage_status)

    report = subparsers.add_parser("report", help="Aggregate local ADD/UPDATE audit logs")
    report.add_argument("--run-id", required=True)
    report.set_defaults(handler=stage_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
