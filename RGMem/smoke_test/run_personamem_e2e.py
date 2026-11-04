"""Evaluate PersonaMem questions through RGMem's original L0+L1 path.

Primary path, aligned with scripts/evaluate_l0_plus_l1.py:
question -> official query-structuring LLM -> L2 relevance retrieval
(L0 relevance + L1 graph) -> official answer-generation prompt.

The only adapter is multiple-choice formatting and exact A/B/C/D scoring.
Gold answers are never passed to query structuring, retrieval, or answering.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SMOKE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SMOKE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = SMOKE_DIR / "artifacts"
CASE_DIR = SMOKE_DIR / "e2e_cases"
SUITES = ("original", "ak", "graph_errors")
RESULT_FILENAME = "rgmem_main_results.jsonl"
ATTRIBUTION_RESULT_FILENAME = "rgmem_attribution_results.jsonl"
ATTRIBUTION_VARIANTS = ("no_memory", "l0_only", "l1_only", "l0_plus_l1")
KANOA_FIRST_TRACE_QUESTION_ID = "original-acd74206-37dc-4756-94a8-b99a395d9a21"

# Retained only for a later attribution experiment. The primary path never reads
# these values when constructing a query.
KANOA_ORIGINAL_ATTRIBUTION_ENTITIES = {
    "acd74206-37dc-4756-94a8-b99a395d9a21": ["music_production", "software_tools", "electronic_music", "pacific_islander_music_fusion"],
    "f546a74f-54de-40d0-9d88-8b0e30467d7b": ["creative_activities", "nature", "sound_engineering", "music_creation"],
    "b3588797-acdf-40d3-bcc5-951f81896f95": ["music_expression", "music_creation", "original_composition", "live_performance", "music_reviews", "podcasting", "music_collaboration"],
    "5370f186-0028-41f4-b436-c48575899fca": ["creative_activities", "emotional_expression", "music_production", "music_creation", "instruments", "playlists"],
    "a40d5f67-8ec6-480b-a901-9709eecee9b9": ["podcasting", "music_trends", "peer_feedback", "content_creation"],
    "6142fba4-a233-405a-a4fe-a5b2e27374a7": ["music_reviews", "album_reviews", "public_criticism", "writing_and_blogging"],
    "2cef53ac-c165-41b5-a31f-c3064106471f": ["music_collaboration", "band_collaboration", "artistic_direction", "creative_autonomy"],
    "4afc48ca-e6a7-4467-abd2-911915199315": ["music_reviews", "album_reviews", "public_criticism", "writing_and_blogging"],
    "defecce2-ff39-438e-ae41-90f3bcf9cb48": ["music_theory", "music_learning", "pacific_islander_music", "cultural_background", "mentorship"],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_dir(run_id: str) -> Path:
    return ARTIFACTS_DIR / f"personamem-{run_id}"


def _eval_dir(run_id: str, eval_id: str) -> Path:
    return _run_dir(run_id) / "end_to_end" / eval_id


def _require_completed_run(run_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = _run_dir(run_id)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Run manifest not found: {manifest_path}")
    missing = [n for n in range(1, 6) if not (run_dir / f"round-{n:02d}" / ".completed").is_file()]
    if missing:
        raise SystemExit(f"Run is incomplete; missing completed rounds: {missing}")
    return run_dir, _read_json(manifest_path)


def _case_sources(run_manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve user-specific diagnostic cases without changing retrieval behavior."""
    user_number = run_manifest.get("dataset_user_number")
    if user_number is None:
        return {
            "ak_path": CASE_DIR / "kanoa_ak_questions.json",
            "graph_errors_path": CASE_DIR / "kanoa_graph_error_questions.json",
            "original_entity_queries": KANOA_ORIGINAL_ATTRIBUTION_ENTITIES,
            "original_entity_queries_path": None,
        }

    user_case_dir = CASE_DIR / "32k" / f"user_{int(user_number):03d}"
    original_entity_queries_path = user_case_dir / "original_entity_queries.json"
    sources = {
        "ak_path": user_case_dir / "ak_questions.json",
        "graph_errors_path": user_case_dir / "graph_error_questions.json",
        "original_entity_queries_path": original_entity_queries_path,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit(
            "User-specific end-to-end case files are incomplete:\n- " + "\n- ".join(missing)
        )
    sources["original_entity_queries"] = _read_json(original_entity_queries_path)
    return sources


def _require_eval(run_id: str, eval_id: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    output_dir = _eval_dir(run_id, eval_id)
    manifest_path = output_dir / "manifest.json"
    questions_path = output_dir / "question_suite.json"
    if not manifest_path.is_file() or not questions_path.is_file():
        raise SystemExit(f"Prepared evaluation not found: {output_dir}\nRun prepare first.")
    return output_dir, _read_json(manifest_path), _read_json(questions_path)


def _normalize_choice(value: Any) -> str:
    match = re.search(r"[A-D]", str(value).upper())
    if not match:
        raise ValueError(f"Cannot normalize answer choice: {value!r}")
    return match.group(0)


def _parse_original_options(value: Any) -> dict[str, str]:
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(str(value))
        except json.JSONDecodeError:
            items = ast.literal_eval(str(value))
    if not isinstance(items, list) or len(items) != 4:
        raise ValueError(f"Expected four options, got: {items!r}")
    return {
        letter: re.sub(r"^\s*\([a-dA-D]\)\s*", "", str(item)).strip()
        for letter, item in zip("ABCD", items)
    }


def _normalize_original_questions(
        rows: list[dict[str, Any]],
        attribution_entities: dict[str, list[str]],
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        source_id = row["question_id"]
        entity_queries = attribution_entities.get(source_id)
        if not entity_queries:
            raise ValueError(f"Missing fixed attribution entity queries for original question: {source_id}")
        normalized.append({
            "question_id": f"original-{source_id}",
            "source_question_id": source_id,
            "suite": "original",
            "question_type": row.get("question_type"),
            "question": row["user_question_or_message"],
            "options": _parse_original_options(row["all_options"]),
            "correct_answer": _normalize_choice(row["correct_answer"]),
            "evidence_indices": [],
            "target_error": "original PersonaMem evaluation item",
            "source_end_index": int(row.get("end_index_in_shared_context", 182)),
            "attribution_entity_queries": entity_queries,
        })
    return normalized


def _normalize_custom_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for source in rows:
        row = dict(source)
        row["attribution_entity_queries"] = row.pop("entity_queries", [])
        normalized.append(row)
    return normalized


def _validate_questions(rows: list[dict[str, Any]]) -> None:
    seen = set()
    for row in rows:
        question_id = row.get("question_id")
        if not question_id or question_id in seen:
            raise ValueError(f"Missing or duplicate question_id: {question_id!r}")
        seen.add(question_id)
        if row.get("suite") not in SUITES:
            raise ValueError(f"Unknown suite for {question_id}: {row.get('suite')!r}")
        if set(row.get("options", {})) != set("ABCD"):
            raise ValueError(f"Question {question_id} must have A/B/C/D options")
        if row.get("correct_answer") not in set("ABCD"):
            raise ValueError(f"Question {question_id} has an invalid correct answer")


def _select_questions(rows: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    return rows if suite == "all" else [row for row in rows if row["suite"] == suite]


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {options[letter]}" for letter in "ABCD")


def _answer_question_text(question: dict[str, Any]) -> str:
    return (
        f"{question['question']}\n\nOptions:\n{_format_options(question['options'])}\n\n"
        "Return exactly one capital letter: A, B, C, or D."
    )


def _answer_question_with_reason_text(question: dict[str, Any]) -> str:
    return (
        f"{question['question']}\n\nOptions:\n{_format_options(question['options'])}\n\n"
        "Return exactly two lines in this format:\n"
        "Choice: A, B, C, or D\n"
        "Reason: explain why you selected that option in no more than 20 English words."
    )


def _parse_model_choice(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().upper()
    exact = re.fullmatch(r"[\(\[]?([A-D])[\)\]\.:]?", text)
    if exact:
        return exact.group(1)
    match = re.search(r"(?:ANSWER|CHOICE)\s*(?:IS|:)?\s*\(?([A-D])\)?", text)
    return match.group(1) if match else None


def _parse_model_reason(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    match = re.search(r"(?is)(?:^|\n)\s*REASON\s*:\s*(.+?)\s*$", raw.strip())
    return match.group(1).strip() if match else None


def _english_word_count(value: str | None) -> int:
    if not value:
        return 0
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protocol_text(persona_name: str, original_question_count: int) -> str:
    return f"""# {persona_name} RGMem primary end-to-end protocol

1. Pass only the question text to RGMem's runtime `QUERY_STRUCTURING_PROMPT`.
2. Use the configured `rag_keyword_model` to generate `entity_queries`.
3. Call original `L2Retriever` with `retrieval_type="relevance"`, matching
   `scripts/evaluate_l0_plus_l1.py`.
4. L2 combines L0 relevance and L1 graph retrieval.
5. Pass the combined context to RGMem's runtime `ANSWER_GENERATION_PROMPT`.
6. Append four options to the question and require A/B/C/D; score exact match.

Gold answers and evidence indices are never passed to the models. Manually chosen
entities are retained only as `attribution_entity_queries`; the primary path never
reads them. All questions use final memory after round 5, so the original
{original_question_count} questions are
final-state diagnostics rather than a reproduction of earlier temporal cutoffs.

The evaluator does not mutate user-memory content. RGMem's original MemoryGraph
initialization may execute `CREATE INDEX ... IF NOT EXISTS`.

The later attribution command uses the same final round-5 memory state and four
paired conditions: no memory, L0 only, L1 only, and L0+L1. It uses the same
manually fixed entity names in all memory conditions, so query-planning variance
cannot be mistaken for a memory-layer effect. For L0 BM25 only, graph-style
underscores are converted to spaces (for example, `audio_production` becomes
`audio production`); L1 continues to receive the original graph entity names.
"""


def stage_prepare(args: argparse.Namespace) -> None:
    run_dir, run_manifest = _require_completed_run(args.run_id)
    output_dir = _eval_dir(args.run_id, args.eval_id)
    if output_dir.exists():
        raise SystemExit(f"Evaluation already exists: {output_dir}\nUse a new --eval-id.")
    case_sources = _case_sources(run_manifest)
    original = _normalize_original_questions(
        _read_json(run_dir / "questions.json"),
        case_sources["original_entity_queries"],
    )
    ak = _normalize_custom_questions(_read_json(case_sources["ak_path"]))
    graph_errors = _normalize_custom_questions(_read_json(case_sources["graph_errors_path"]))
    combined = original + ak + graph_errors
    _validate_questions(combined)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "questions_original.json", original)
    _write_json(output_dir / "questions_ak.json", ak)
    _write_json(output_dir / "questions_graph_errors.json", graph_errors)
    _write_json(output_dir / "question_suite.json", combined)
    persona_name = str(run_manifest.get("persona_name") or run_manifest["user_id"])
    (output_dir / "protocol.md").write_text(
        _protocol_text(persona_name, len(original)), encoding="utf-8"
    )
    _write_json(output_dir / "manifest.json", {
        "eval_id": args.eval_id,
        "source_run_id": args.run_id,
        "user_id": run_manifest["user_id"],
        "role_id": int(run_manifest["role_id"]),
        "shared_context_id": run_manifest.get("shared_context_id"),
        "dataset_user_number": run_manifest.get("dataset_user_number"),
        "persona_name": persona_name,
        "created_at": _utc_now(),
        "memory_state": "final_after_round_5_and_consolidation_5",
        "first_original_question_id": original[0]["question_id"],
        "case_sources": {
            "ak": str(case_sources["ak_path"]),
            "graph_errors": str(case_sources["graph_errors_path"]),
            "original_entity_queries": (
                str(case_sources["original_entity_queries_path"])
                if case_sources["original_entity_queries_path"] is not None
                else "built-in Kanoa legacy mapping"
            ),
        },
        "question_counts": {"original": len(original), "ak": len(ak), "graph_errors": len(graph_errors), "all": len(combined)},
        "primary_retrieval": "RGMem L2 relevance (L0 relevance + L1 graph)",
        "query_planning": "official configured query-structuring LLM",
        "scoring": "exact multiple-choice letter; no LLM judge",
        "user_memory_mutations": False,
        "memory_graph_initialization_may_ensure_indexes": True,
    })
    print(f"Prepared RGMem end-to-end evaluation: {output_dir}")
    print(f"Questions: original={len(original)}, ak={len(ak)}, graph_errors={len(graph_errors)}, all={len(combined)}")
    print("No API calls or database connections were made.")


def _initialize_official_pipeline() -> dict[str, Any]:
    from scripts.evaluate_l0_plus_l1 import (
        ANSWER_GENERATION_PROMPT,
        QUERY_STRUCTURING_PROMPT,
        sanitize_llm_json_output,
    )
    from config import settings
    from l0_memory.llm_interface.factory import LlmFactory as L0LlmFactory
    from l1_memory.configs.base import MemoryConfig
    from l1_memory.graphs.neo4j import MemoryGraph
    from l1_memory.memory.main import Memory
    from l2_retrieval import L2Retriever
    from l2_retrieval.l1_adapter import L1Adapter

    l1_config = MemoryConfig(llm=settings.L1_LLM_CONFIG, embedder=settings.L1_EMBEDDER_CONFIG, graph_store=settings.L1_GRAPH_STORE_CONFIG)
    graph = MemoryGraph(config=l1_config)
    l2 = L2Retriever(l1_adapter=L1Adapter(l1_memory=Memory(memory_graph=graph)))
    provider = settings.LLM_PROVIDER
    cfg = settings.L0_LLM_CONFIGS.get(provider)
    if not cfg:
        raise ValueError(f"Missing L0 provider configuration: {provider}")
    query_model = cfg["rag_keyword_model"]
    answer_model = cfg["task_models"]["eval_chat_model"]
    base = {"api_key": cfg.get("api_key"), "base_url": cfg.get("base_url"), **cfg.get("params", {})}
    query_client = L0LlmFactory.create(provider, {"model": query_model, **base}, "PersonaMemOfficialQueryStructClient")
    answer_client = L0LlmFactory.create(provider, {"model": answer_model, **base}, "PersonaMemOfficialEvalChatClient")
    return {
        "query_prompt": QUERY_STRUCTURING_PROMPT,
        "answer_prompt": ANSWER_GENERATION_PROMPT,
        "sanitize": sanitize_llm_json_output,
        "l2": l2,
        "query_client": query_client,
        "answer_client": answer_client,
        "query_model": query_model,
        "answer_model": answer_model,
    }


def _official_parse_entity_queries(raw: Any, sanitize) -> list[dict[str, Any]]:
    parsed: Any = []
    response_to_parse = ""
    if isinstance(raw, list):
        parsed = raw
    elif isinstance(raw, str):
        response_to_parse = raw
    elif isinstance(raw, dict):
        response_to_parse = json.dumps(raw, ensure_ascii=False)
    if response_to_parse:
        try:
            parsed = json.loads(sanitize(response_to_parse))
        except json.JSONDecodeError:
            parsed = []
    entity_queries = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "name" in item:
                entity_queries.append(item)
            elif isinstance(item, str):
                entity_queries.append({"name": item, "retrieve": []})
    return entity_queries


def stage_evaluate(args: argparse.Namespace) -> None:
    if not args.confirm_api_and_read:
        raise SystemExit("This calls LLM/embedding APIs and reads L0/L1. Re-run with --confirm-api-and-read.")
    output_dir, manifest, questions = _require_eval(args.run_id, args.eval_id)
    selected = _select_questions(questions, args.suite)
    result_path = output_dir / RESULT_FILENAME
    completed_ids = {row["question_id"] for row in _read_jsonl(result_path) if row.get("status") == "completed"}
    pipeline = _initialize_official_pipeline()
    completed_now = skipped = 0

    for question in selected:
        qid = question["question_id"]
        if qid in completed_ids:
            skipped += 1
            continue
        base_row = {
            "question_id": qid,
            "suite": question["suite"],
            "started_at": _utc_now(),
            "query_model": pipeline["query_model"],
            "answer_model": pipeline["answer_model"],
            "retrieval_type": "relevance",
        }
        try:
            struct_prompt = pipeline["query_prompt"].format(query=question["question"])
            structured_raw = pipeline["query_client"].get_structured_completion(struct_prompt, model_override=pipeline["query_model"])
            entity_queries = _official_parse_entity_queries(structured_raw, pipeline["sanitize"])
            context = asyncio.run(pipeline["l2"].retrieve(
                query=question["question"],
                user_id=str(manifest["user_id"]),
                role_id=int(manifest["role_id"]),
                retrieval_type="relevance",
                entity_queries=entity_queries,
            ))
            answer_prompt = pipeline["answer_prompt"].format(context=context, question=_answer_question_text(question))
            raw_answer = pipeline["answer_client"].get_completion(answer_prompt, model_override=pipeline["answer_model"])
            choice = _parse_model_choice(raw_answer)
            common = {
                **base_row,
                "structured_query_raw": structured_raw,
                "entity_queries": entity_queries,
                "query_plan_empty": not bool(entity_queries),
                "retrieved_context": context,
                "raw_answer": raw_answer,
            }
            if choice is None:
                row = {**common, "status": "failed", "failed_at": _utc_now(), "error_type": "EMPTY_OR_UNPARSEABLE_ANSWER"}
            else:
                row = {
                    **common,
                    "status": "completed",
                    "completed_at": _utc_now(),
                    "question": question["question"],
                    "options": question["options"],
                    "correct_answer": question["correct_answer"],
                    "retrieved_context_chars": len(context),
                    "choice": choice,
                    "correct": choice == question["correct_answer"],
                }
                completed_now += 1
                completed_ids.add(qid)
        except Exception as exc:
            row = {**base_row, "status": "failed", "failed_at": _utc_now(), "error_type": type(exc).__name__, "error": str(exc)}
        _append_jsonl(result_path, row)
        print(f"[{row['status']}] {qid}")
    print(f"RGMem primary evaluation finished: requested={len(selected)}, completed_now={completed_now}, skipped_existing={skipped}")
    print(f"Results: {result_path}")


def _attribution_queries(question: dict[str, Any]) -> list[dict[str, Any]]:
    names = question.get("attribution_entity_queries") or []
    normalized = []
    for item in names:
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str) and name.strip():
            normalized.append({
                "name": name.strip(),
                "retrieve": ["history", "summary", "inference"],
            })
    if not normalized:
        raise ValueError(f"No attribution entity queries for {question.get('question_id')}")
    return normalized


def _attribution_l0_keywords(entity_queries: list[dict[str, Any]]) -> str:
    """Convert graph-style entity names into natural text for L0 BM25."""
    return " ".join(item["name"].replace("_", " ") for item in entity_queries)


async def _retrieve_attribution_context(
        pipeline: dict[str, Any], manifest: dict[str, Any], question: dict[str, Any],
        variant: str, entity_queries: list[dict[str, Any]]) -> str:
    from l2_retrieval.formatter import format_context_as_markdown

    query = question["question"]
    user_id = str(manifest["user_id"])
    role_id = int(manifest["role_id"])
    if variant == "no_memory":
        return format_context_as_markdown(None, None, None, query)
    if variant == "l0_only":
        keywords = _attribution_l0_keywords(entity_queries)
        return await pipeline["l2"].retrieve(
            query=query, user_id=user_id, role_id=role_id,
            retrieval_type="relevance", keywords=keywords,
        )
    if variant == "l1_only":
        l1_data = await asyncio.to_thread(
            pipeline["l2"].l1_adapter.retrieve, user_id, role_id, entity_queries,
        )
        return format_context_as_markdown(None, None, l1_data, query)
    if variant == "l0_plus_l1":
        keywords = _attribution_l0_keywords(entity_queries)
        l0_task = asyncio.to_thread(
            pipeline["l2"].l0_adapter.retrieve_relevance,
            user_id, role_id, query, keywords,
        )
        l1_task = asyncio.to_thread(
            pipeline["l2"].l1_adapter.retrieve,
            user_id, role_id, entity_queries,
        )
        l0_data, l1_data = await asyncio.gather(l0_task, l1_task)
        return format_context_as_markdown(None, l0_data, l1_data, query)
    raise ValueError(f"Unknown attribution variant: {variant}")


def stage_attribute(args: argparse.Namespace) -> None:
    if not args.confirm_api_and_read:
        raise SystemExit("This calls LLM/embedding APIs and reads L0/L1. Re-run with --confirm-api-and-read.")
    output_dir, manifest, questions = _require_eval(args.run_id, args.eval_id)
    selected = _select_questions(questions, args.suite)
    selected_variants = list(ATTRIBUTION_VARIANTS) if args.variant == "all" else [args.variant]
    result_path = output_dir / ATTRIBUTION_RESULT_FILENAME
    completed_keys = {
        (row.get("question_id"), row.get("variant"))
        for row in _read_jsonl(result_path) if row.get("status") == "completed"
    }
    pipeline = _initialize_official_pipeline()
    completed_now = skipped = 0

    for question in selected:
        qid = question["question_id"]
        entity_queries = _attribution_queries(question)
        for variant in selected_variants:
            key = (qid, variant)
            if key in completed_keys:
                skipped += 1
                continue
            base_row = {
                "question_id": qid,
                "suite": question["suite"],
                "variant": variant,
                "started_at": _utc_now(),
                "answer_model": pipeline["answer_model"],
                "query_source": "fixed_attribution_entity_queries",
                "entity_queries": entity_queries,
            }
            try:
                context = asyncio.run(_retrieve_attribution_context(
                    pipeline, manifest, question, variant, entity_queries,
                ))
                answer_prompt = pipeline["answer_prompt"].format(
                    context=context, question=_answer_question_text(question),
                )
                raw_answer = pipeline["answer_client"].get_completion(
                    answer_prompt, model_override=pipeline["answer_model"],
                )
                choice = _parse_model_choice(raw_answer)
                common = {
                    **base_row,
                    "retrieved_context": context,
                    "retrieved_context_chars": len(context),
                    "raw_answer": raw_answer,
                }
                if choice is None:
                    row = {
                        **common, "status": "failed", "failed_at": _utc_now(),
                        "error_type": "EMPTY_OR_UNPARSEABLE_ANSWER",
                    }
                else:
                    row = {
                        **common,
                        "status": "completed",
                        "completed_at": _utc_now(),
                        "question": question["question"],
                        "options": question["options"],
                        "correct_answer": question["correct_answer"],
                        "choice": choice,
                        "correct": choice == question["correct_answer"],
                    }
                    completed_now += 1
                    completed_keys.add(key)
            except Exception as exc:
                row = {
                    **base_row, "status": "failed", "failed_at": _utc_now(),
                    "error_type": type(exc).__name__, "error": str(exc),
                }
            _append_jsonl(result_path, row)
            print(f"[{row['status']}] {qid} / {variant}")

    requested = len(selected) * len(selected_variants)
    print(
        f"RGMem attribution finished: variants={','.join(selected_variants)}, "
        f"requested={requested}, completed_now={completed_now}, skipped_existing={skipped}"
    )
    print(f"Results: {result_path}")


def _trace_checkpoint(trace_dir: Path, event: str, **details: Any) -> None:
    _append_jsonl(trace_dir / "checkpoints.jsonl", {
        "timestamp": _utc_now(),
        "event": event,
        **details,
    })


def _answer_request_metadata(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Describe the effective answer request without exposing credentials."""
    model = str(pipeline["answer_model"])
    config = pipeline["answer_client"].config
    parameters: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "See the saved exact prompt file."}],
    }
    omitted_parameters: list[str] = []
    if model.lower().startswith("gpt-6"):
        parameters["max_completion_tokens"] = config.max_tokens
        omitted_parameters = ["temperature", "top_p", "max_tokens"]
    else:
        parameters.update({
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        })
    return {
        "provider": "openai",
        "endpoint_style": "chat.completions",
        "effective_parameters": parameters,
        "omitted_parameters": omitted_parameters,
        "stateless_request": True,
        "conversation_history_sent": False,
    }


def _l0_result_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {"core_facts": 0, "conclusions": 0}
    return {
        "core_facts": len(raw.get("core_facts") or []),
        "conclusions": len(raw.get("conclusions") or []),
    }


def _l1_result_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict) or raw.get("error"):
        return {"entities": 0, "summaries": 0, "inferences": 0, "one_hop_details": 0}
    return {
        "entities": len(raw),
        "summaries": sum(len(item.get("summaries") or []) for item in raw.values()),
        "inferences": sum(len(item.get("inferences") or []) for item in raw.values()),
        "one_hop_details": sum(len(item.get("one_hop_details") or []) for item in raw.values()),
    }


def stage_trace_first(args: argparse.Namespace) -> None:
    """Trace only the first PersonaMem item through four controlled memory conditions."""
    if not args.confirm_api_and_read:
        raise SystemExit("This calls embedding/LLM APIs and reads L0/L1. Re-run with --confirm-api-and-read.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.trace_id):
        raise SystemExit("--trace-id may contain only letters, numbers, dot, underscore, and hyphen.")

    output_dir, manifest, questions = _require_eval(args.run_id, args.eval_id)
    first_question_id = manifest.get("first_original_question_id")
    if not first_question_id:
        first_question_id = next(
            (item["question_id"] for item in questions if item.get("suite") == "original"),
            KANOA_FIRST_TRACE_QUESTION_ID,
        )
    question = next((item for item in questions if item["question_id"] == first_question_id), None)
    if question is None:
        raise SystemExit(f"The first original question is missing: {first_question_id}")

    trace_dir = output_dir / "first_question_traces" / args.trace_id
    completed_marker = trace_dir / ".completed"
    if completed_marker.exists():
        raise SystemExit(f"Trace is already completed: {trace_dir}\nUse a new --trace-id for another model sample.")
    trace_dir.mkdir(parents=True, exist_ok=True)
    started_marker = trace_dir / ".started"
    if not started_marker.exists():
        _write_text(started_marker, _utc_now() + "\n")
        _trace_checkpoint(trace_dir, "trace_started")

    model_question = {
        "question_id": question["question_id"],
        "question": question["question"],
        "options": question["options"],
    }
    scoring_reference = {
        "question_id": question["question_id"],
        "correct_answer": question["correct_answer"],
        "note": "Stored for post-hoc exact-match scoring; never inserted into a model prompt.",
    }
    entity_queries = _attribution_queries(question)
    _write_json(trace_dir / "00_trace_manifest.json", {
        "trace_id": args.trace_id,
        "source_eval_id": args.eval_id,
        "source_run_id": args.run_id,
        "memory_state": manifest.get("memory_state"),
        "question_policy": "exactly one fixed first PersonaMem question",
        "question_id": first_question_id,
        "variants": list(ATTRIBUTION_VARIANTS),
        "gold_answer_sent_to_model": False,
        "user_memory_mutations": False,
    })
    _write_json(trace_dir / "01_question_for_model.json", model_question)
    _write_json(trace_dir / "01_scoring_reference_not_for_model.json", scoring_reference)
    _write_json(trace_dir / "02_fixed_retrieval_request.json", {
        "query": question["question"],
        "user_id": str(manifest["user_id"]),
        "role_id": int(manifest["role_id"]),
        "retrieval_type": "relevance",
        "entity_queries": entity_queries,
        "query_source": "manually fixed diagnostic control; no query-structuring LLM is called",
    })
    _write_json(trace_dir / "02_code_provenance.json", {
        "l0_generation": {
            "meaning": "L0 is distilled memory generated earlier from raw dialogue, not raw dialogue text itself.",
            "source": "l0_memory/core/memory_processor.py -> Elasticsearch core_fact/conclusions",
        },
        "l0_retrieval": {
            "source": "l2_retrieval/l0_adapter.py:L0Adapter.retrieve_relevance -> l0_memory/rag_system/rag_server.py:RagServer.get_relevance_context",
            "method": "RGMem Elasticsearch keyword/vector/hybrid relevance retrieval",
            "diagnostic_query_adaptation": "Graph-style underscores are replaced with spaces for L0 text search only.",
        },
        "l1_retrieval": {
            "source": "l2_retrieval/l1_adapter.py:L1Adapter.retrieve -> l1_memory/graphs/graph_retrieval.py:GraphRetriever.extract_knowledge",
            "method": "RGMem cosine-similarity node matching plus summaries, inferences, and one-hop relationship evidence",
        },
        "l2_aggregation": {
            "source": "l2_retrieval/retriever.py:L2Retriever.retrieve and l2_retrieval/formatter.py:format_context_as_markdown",
            "method": "RGMem combines formatted L0 and L1 evidence; this trace reuses those same adapters and formatter while exposing each raw layer separately.",
        },
        "synergy_and_correction": {
            "source": "l1_memory/graphs/utils.py:UPDATE_HIERARCHICAL_SUMMARY_PROMPT",
            "meaning": "Paper-level order/correction information is materialized inside summary text as Macro-Law, Core User Traits (Synergy), and Action/Recommendation Constraints (Tension).",
            "retrieval_behavior": "No extra hidden score is computed at answer time; these terms affect the answer only when a retrieved L1 summary containing them is placed in the final prompt.",
        },
    })

    _trace_checkpoint(trace_dir, "pipeline_initialization_started")
    pipeline = _initialize_official_pipeline()
    _write_json(trace_dir / "03_answer_api_request_metadata.json", _answer_request_metadata(pipeline))
    _trace_checkpoint(
        trace_dir, "pipeline_initialization_completed",
        answer_model=pipeline["answer_model"],
        query_model_configured_but_not_called=pipeline["query_model"],
    )

    query = question["question"]
    user_id = str(manifest["user_id"])
    role_id = int(manifest["role_id"])
    keywords = _attribution_l0_keywords(entity_queries)
    _write_json(trace_dir / "04_effective_retrieval_inputs.json", {
        "l0": {"query": query, "keywords": keywords, "user_id": user_id, "role_id": role_id},
        "l1": {"entity_queries": entity_queries, "user_id": user_id, "role_id": role_id},
    })

    l0_path = trace_dir / "05_l0_raw_retrieval.json"
    if l0_path.exists():
        l0_raw = _read_json(l0_path)
        _trace_checkpoint(trace_dir, "l0_raw_reused_from_checkpoint", counts=_l0_result_counts(l0_raw))
    else:
        _trace_checkpoint(trace_dir, "l0_retrieval_started")
        l0_raw = pipeline["l2"].l0_adapter.retrieve_relevance(user_id, role_id, query, keywords)
        _write_json(l0_path, l0_raw)
        _trace_checkpoint(trace_dir, "l0_retrieval_completed", counts=_l0_result_counts(l0_raw))

    l1_path = trace_dir / "06_l1_raw_retrieval.json"
    if l1_path.exists():
        l1_raw = _read_json(l1_path)
        _trace_checkpoint(trace_dir, "l1_raw_reused_from_checkpoint", counts=_l1_result_counts(l1_raw))
    else:
        _trace_checkpoint(trace_dir, "l1_retrieval_started")
        l1_raw = pipeline["l2"].l1_adapter.retrieve(user_id, role_id, entity_queries)
        _write_json(l1_path, l1_raw)
        _trace_checkpoint(trace_dir, "l1_retrieval_completed", counts=_l1_result_counts(l1_raw))

    from l2_retrieval.formatter import format_context_as_markdown

    contexts = {
        "no_memory": format_context_as_markdown(None, None, None, query),
        "l0_only": format_context_as_markdown(None, l0_raw, None, query),
        "l1_only": format_context_as_markdown(None, None, l1_raw, query),
        "l0_plus_l1": format_context_as_markdown(None, l0_raw, l1_raw, query),
    }
    _write_json(trace_dir / "07_context_comparisons.json", {
        "context_characters": {name: len(value) for name, value in contexts.items()},
        "context_sha256": {name: _sha256_text(value) for name, value in contexts.items()},
        "no_memory_equals_l0_only": contexts["no_memory"] == contexts["l0_only"],
        "l1_only_equals_l0_plus_l1": contexts["l1_only"] == contexts["l0_plus_l1"],
        "l1_contains_macro_law": "Macro-Law" in contexts["l1_only"],
        "l1_contains_synergy": "Synergy" in contexts["l1_only"],
        "l1_contains_tension": "Tension" in contexts["l1_only"],
    })

    results = []
    for variant in ATTRIBUTION_VARIANTS:
        variant_dir = trace_dir / "variants" / variant
        context = contexts[variant]
        context_path = variant_dir / "01_formatted_context.md"
        prompt_path = variant_dir / "02_exact_answer_prompt.txt"
        output_path = variant_dir / "03_model_raw_output.txt"
        result_path = variant_dir / "04_parsed_result.json"
        _write_text(context_path, context + "\n")
        answer_prompt = pipeline["answer_prompt"].format(
            context=context,
            question=_answer_question_with_reason_text(question),
        )
        _write_text(prompt_path, answer_prompt)
        _write_json(variant_dir / "02_api_request_metadata.json", {
            **_answer_request_metadata(pipeline),
            "exact_prompt_file": str(prompt_path),
            "prompt_sha256": _sha256_text(answer_prompt),
            "gold_answer_sent_to_model": False,
        })
        _trace_checkpoint(trace_dir, "answer_prompt_saved", variant=variant, prompt_sha256=_sha256_text(answer_prompt))

        if output_path.exists():
            raw_answer = output_path.read_text(encoding="utf-8")
            _trace_checkpoint(trace_dir, "model_output_reused_from_checkpoint", variant=variant)
        else:
            _trace_checkpoint(trace_dir, "model_call_started", variant=variant, model=pipeline["answer_model"])
            raw_value = pipeline["answer_client"].get_completion(
                answer_prompt, model_override=pipeline["answer_model"],
            )
            raw_answer = raw_value if isinstance(raw_value, str) else json.dumps(raw_value, ensure_ascii=False)
            _write_text(output_path, raw_answer)
            _trace_checkpoint(trace_dir, "model_call_completed", variant=variant)

        choice = _parse_model_choice(raw_answer)
        reason = _parse_model_reason(raw_answer)
        reason_words = _english_word_count(reason)
        if choice is None:
            status = "failed"
            error_type = "EMPTY_OR_UNPARSEABLE_CHOICE"
        elif reason is None:
            status = "failed"
            error_type = "EMPTY_OR_UNPARSEABLE_REASON"
        elif reason_words > 20:
            status = "failed"
            error_type = "REASON_EXCEEDS_20_ENGLISH_WORDS"
        else:
            status = "completed"
            error_type = None
        result = {
            "status": status,
            "error_type": error_type,
            "question_id": question["question_id"],
            "variant": variant,
            "answer_model": pipeline["answer_model"],
            "context_chars": len(context),
            "context_sha256": _sha256_text(context),
            "prompt_sha256": _sha256_text(answer_prompt),
            "choice": choice,
            "reason": reason,
            "reason_english_word_count": reason_words,
            "reason_within_20_english_words": reason is not None and reason_words <= 20,
            "correct_answer": question["correct_answer"],
            "correct": choice == question["correct_answer"] if choice else None,
            "gold_answer_sent_to_model": False,
            "completed_at": _utc_now(),
        }
        _write_json(result_path, result)
        _trace_checkpoint(trace_dir, "variant_result_saved", variant=variant, status=status, choice=choice)
        results.append(result)

    report = {
        "trace_id": args.trace_id,
        "question_id": question["question_id"],
        "answer_model": pipeline["answer_model"],
        "l0_counts": _l0_result_counts(l0_raw),
        "l1_counts": _l1_result_counts(l1_raw),
        "context_comparisons": _read_json(trace_dir / "07_context_comparisons.json"),
        "results": results,
        "interpretation_guardrail": (
            "If no_memory is correct, that alone is not evidence of hidden API history. "
            "Its saved prompt contains only the question/options and the literal no-memory context. "
            "A correct answer can arise from option wording, model priors, or generic-answer bias."
        ),
    }
    _write_json(trace_dir / "08_trace_report.json", report)
    _write_text(completed_marker, _utc_now() + "\n")
    _trace_checkpoint(trace_dir, "trace_completed")
    print(f"First-question trace completed: {trace_dir}")
    for result in results:
        print(
            f"{result['variant']}: status={result['status']}; choice={result['choice']}; "
            f"reason_words={result['reason_english_word_count']}"
        )


def _latest_attribution_completed(
        rows: list[dict[str, Any]], question_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {}
    for row in rows:
        key = (row.get("question_id"), row.get("variant"))
        if row.get("status") == "completed" and key[0] in question_ids and key[1] in ATTRIBUTION_VARIANTS:
            latest[key] = row
    return latest


def stage_attribution_status(args: argparse.Namespace) -> None:
    output_dir, manifest, questions = _require_eval(args.run_id, args.eval_id)
    question_ids = {q["question_id"] for q in questions}
    rows = _read_jsonl(output_dir / ATTRIBUTION_RESULT_FILENAME)
    completed = _latest_attribution_completed(rows, question_ids)
    print(f"Attribution: {manifest['eval_id']}")
    print(f"Source run: {manifest['source_run_id']}")
    for variant in ATTRIBUTION_VARIANTS:
        count = sum((qid, variant) in completed for qid in question_ids)
        print(f"{variant}: completed={count}/{len(question_ids)}")
    print(f"Directory: {output_dir}")


def stage_attribution_report(args: argparse.Namespace) -> None:
    output_dir, manifest, questions = _require_eval(args.run_id, args.eval_id)
    qmap = {q["question_id"]: q for q in questions}
    rows = _read_jsonl(output_dir / ATTRIBUTION_RESULT_FILENAME)
    completed = _latest_attribution_completed(rows, set(qmap))
    stats = defaultdict(lambda: {"completed": 0, "correct": 0})
    suite_stats = defaultdict(lambda: {"completed": 0, "correct": 0})
    for (qid, variant), row in completed.items():
        stats[variant]["completed"] += 1
        stats[variant]["correct"] += int(row["correct"])
        suite_stats[(row["suite"], variant)]["completed"] += 1
        suite_stats[(row["suite"], variant)]["correct"] += int(row["correct"])

    lines = [
        "# Kanoa RGMem x4 attribution report", "",
        f"- Evaluation: `{manifest['eval_id']}`",
        f"- Source run: `{manifest['source_run_id']}`",
        "- Conditions: no memory / L0 only / L1 only / L0+L1",
        "- Query control: fixed entity queries shared by all memory conditions",
        "- Scoring: exact A/B/C/D", "", "## Overall accuracy", "",
        "| Variant | Correct | Completed | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for variant in ATTRIBUTION_VARIANTS:
        s = stats[variant]
        acc = 100 * s["correct"] / s["completed"] if s["completed"] else 0.0
        lines.append(f"| {variant} | {s['correct']} | {s['completed']} | {acc:.2f}% |")
    lines += ["", "## Accuracy by suite", "", "| Suite | Variant | Correct | Completed | Accuracy |", "|---|---|---:|---:|---:|"]
    for suite in SUITES:
        for variant in ATTRIBUTION_VARIANTS:
            s = suite_stats[(suite, variant)]
            acc = 100 * s["correct"] / s["completed"] if s["completed"] else 0.0
            lines.append(f"| {suite} | {variant} | {s['correct']} | {s['completed']} | {acc:.2f}% |")
    lines += ["", "## Paired per-question outcomes", "", "| Question | Suite | Gold | No memory | L0 | L1 | L0+L1 |", "|---|---|---:|---:|---:|---:|---:|"]
    per_question = []
    for question in questions:
        qid = question["question_id"]
        outcomes = {}
        for variant in ATTRIBUTION_VARIANTS:
            row = completed.get((qid, variant))
            outcomes[variant] = {
                "choice": row.get("choice") if row else None,
                "correct": row.get("correct") if row else None,
            }
        cells = []
        for variant in ATTRIBUTION_VARIANTS:
            outcome = outcomes[variant]
            cells.append(outcome["choice"] if outcome["choice"] else "—")
        lines.append(f"| {qid} | {question['suite']} | {question['correct_answer']} | {' | '.join(cells)} |")
        per_question.append({
            "question_id": qid, "suite": question["suite"],
            "correct_answer": question["correct_answer"], "outcomes": outcomes,
        })

    unresolved = [
        row for row in rows
        if row.get("status") == "failed"
        and (row.get("question_id"), row.get("variant")) not in completed
    ]
    report_json = {
        "eval_id": manifest["eval_id"],
        "source_run_id": manifest["source_run_id"],
        "generated_at": _utc_now(),
        "variants": list(ATTRIBUTION_VARIANTS),
        "overall_stats": dict(stats),
        "suite_stats": {f"{suite}/{variant}": value for (suite, variant), value in suite_stats.items()},
        "per_question": per_question,
        "unresolved_failures": unresolved,
    }
    (output_dir / "rgmem_attribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output_dir / "rgmem_attribution_report.json", report_json)
    print(f"JSON report: {output_dir / 'rgmem_attribution_report.json'}")
    print(f"Markdown report: {output_dir / 'rgmem_attribution_report.md'}")


def _latest_completed(rows: list[dict[str, Any]], ids: set[str]) -> dict[str, dict[str, Any]]:
    latest = {}
    for row in rows:
        if row.get("status") == "completed" and row.get("question_id") in ids:
            latest[row["question_id"]] = row
    return latest


def stage_status(args: argparse.Namespace) -> None:
    output_dir, manifest, questions = _require_eval(args.run_id, args.eval_id)
    rows = _read_jsonl(output_dir / RESULT_FILENAME)
    ids = {q["question_id"] for q in questions}
    completed = _latest_completed(rows, ids)
    failed_ids = {row["question_id"] for row in rows if row.get("status") == "failed" and row.get("question_id") not in completed}
    print(f"Evaluation: {manifest['eval_id']}")
    print(f"Source run: {manifest['source_run_id']}")
    for suite in SUITES:
        suite_ids = {q["question_id"] for q in questions if q["suite"] == suite}
        print(f"{suite}: completed={len(suite_ids & completed.keys())}/{len(suite_ids)}; failed_without_success={len(suite_ids & failed_ids)}")
    print(f"Directory: {output_dir}")


def stage_report(args: argparse.Namespace) -> None:
    output_dir, manifest, questions = _require_eval(args.run_id, args.eval_id)
    qmap = {q["question_id"]: q for q in questions}
    rows = _read_jsonl(output_dir / RESULT_FILENAME)
    completed = _latest_completed(rows, set(qmap))
    failures = [row for row in rows if row.get("status") == "failed" and row.get("question_id") not in completed]
    stats = defaultdict(lambda: {"planned": 0, "completed": 0, "correct": 0})
    for q in questions:
        stats[q["suite"]]["planned"] += 1
    for row in completed.values():
        stats[row["suite"]]["completed"] += 1
        stats[row["suite"]]["correct"] += int(row["correct"])

    lines = [
        "# Kanoa RGMem primary end-to-end report", "",
        f"- Evaluation: `{manifest['eval_id']}`", f"- Source run: `{manifest['source_run_id']}`",
        "- Retrieval: official query structuring + L2 relevance (L0 relevance + L1)",
        "- Scoring: exact A/B/C/D", "", "## Accuracy", "",
        "| Suite | Correct | Completed | Planned | Accuracy over completed |",
        "|---|---:|---:|---:|---:|",
    ]
    for suite in SUITES:
        s = stats[suite]
        acc = s["correct"] / s["completed"] * 100 if s["completed"] else 0.0
        lines.append(f"| {suite} | {s['correct']} | {s['completed']} | {s['planned']} | {acc:.2f}% |")
    lines += ["", "## Per-question results", "", "| Question | Suite | Gold | Answer | Correct | Empty query plan |", "|---|---|---:|---:|---:|---:|"]
    report_rows = []
    for q in questions:
        result = completed.get(q["question_id"])
        answer = result["choice"] if result else "—"
        correct = ("yes" if result["correct"] else "no") if result else "—"
        empty = ("yes" if result.get("query_plan_empty") else "no") if result else "—"
        lines.append(f"| {q['question_id']} | {q['suite']} | {q['correct_answer']} | {answer} | {correct} | {empty} |")
        report_rows.append({
            "question_id": q["question_id"], "suite": q["suite"], "correct_answer": q["correct_answer"],
            "choice": result.get("choice") if result else None,
            "correct": result.get("correct") if result else None,
            "query_plan_empty": result.get("query_plan_empty") if result else None,
        })
    lines += ["", "## Engineering status", "", f"- Unresolved failed questions: {len(failures)}", "- Raw query plans, contexts, and answers: `rgmem_main_results.jsonl`.", "- Wrong answers are not attributed to L0 or L1 in this primary report."]
    (output_dir / "rgmem_main_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output_dir / "rgmem_main_report.json", {
        "eval_id": manifest["eval_id"], "source_run_id": manifest["source_run_id"], "generated_at": _utc_now(),
        "retrieval": "official query structuring + L2 relevance (L0 relevance + L1)",
        "suite_stats": dict(stats), "per_question": report_rows, "unresolved_failures": failures,
    })
    print(f"JSON report: {output_dir / 'rgmem_main_report.json'}")
    print(f"Markdown report: {output_dir / 'rgmem_main_report.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PersonaMem evaluation through RGMem's original L0+L1 relevance path")
    subs = parser.add_subparsers(dest="command", required=True)
    prepare = subs.add_parser("prepare", help="Prepare 27 local questions; no API/database access")
    prepare.add_argument("--run-id", required=True); prepare.add_argument("--eval-id", required=True); prepare.set_defaults(func=stage_prepare)
    evaluate = subs.add_parser("evaluate", help="Run official query structuring, L2 relevance, and answering")
    evaluate.add_argument("--run-id", required=True); evaluate.add_argument("--eval-id", required=True)
    evaluate.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    evaluate.add_argument("--confirm-api-and-read", action="store_true"); evaluate.set_defaults(func=stage_evaluate)
    status = subs.add_parser("status", help="Show local completion status")
    status.add_argument("--run-id", required=True); status.add_argument("--eval-id", required=True); status.set_defaults(func=stage_status)
    report = subs.add_parser("report", help="Build primary-result reports")
    report.add_argument("--run-id", required=True); report.add_argument("--eval-id", required=True); report.set_defaults(func=stage_report)
    attribute = subs.add_parser("attribute", help="Run paired no-memory/L0/L1/L0+L1 attribution")
    attribute.add_argument("--run-id", required=True); attribute.add_argument("--eval-id", required=True)
    attribute.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    attribute.add_argument("--variant", choices=(*ATTRIBUTION_VARIANTS, "all"), default="all")
    attribute.add_argument("--confirm-api-and-read", action="store_true"); attribute.set_defaults(func=stage_attribute)
    trace_first = subs.add_parser(
        "trace-first",
        help="Trace only the fixed first question with raw L0, raw L1, exact prompts, and <=20-word reasons",
    )
    trace_first.add_argument("--run-id", required=True); trace_first.add_argument("--eval-id", required=True)
    trace_first.add_argument("--trace-id", required=True)
    trace_first.add_argument("--confirm-api-and-read", action="store_true"); trace_first.set_defaults(func=stage_trace_first)
    attribution_status = subs.add_parser("attribution-status", help="Show x4 attribution completion status")
    attribution_status.add_argument("--run-id", required=True); attribution_status.add_argument("--eval-id", required=True)
    attribution_status.set_defaults(func=stage_attribution_status)
    attribution_report = subs.add_parser("attribution-report", help="Build x4 attribution reports")
    attribution_report.add_argument("--run-id", required=True); attribution_report.add_argument("--eval-id", required=True)
    attribution_report.set_defaults(func=stage_attribution_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
