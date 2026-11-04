"""Ablate memory layers from an already completed RGMem primary evaluation.

This is deliberately a thin smoke-test wrapper. It does not plan queries and it
does not call Elasticsearch, Neo4j, L0Adapter, L1Adapter, or L2Retriever.
Instead, it reads the exact ``retrieved_context`` saved by the original RGMem
primary path and separates the L0 and L1 blocks using the headings emitted by
``l2_retrieval.formatter``. The original combined context is replayed byte for
byte for the L0+L1 condition.

The only API call made by ``evaluate`` is the final answer-model call.
"""

from __future__ import annotations

import argparse
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
SOURCE_RESULT_FILENAME = "rgmem_main_results.jsonl"
SLICE_FILENAME = "official_context_slices.jsonl"
RESULT_FILENAME = "official_context_ablation_results.jsonl"
VARIANTS = ("no_memory", "l0_only", "l1_only", "l0_plus_l1")
SUITES = ("original", "ak", "graph_errors")
NO_MEMORY_CONTEXT = "No relevant information found in memory."
L0_HEADINGS = ("## Recent Facts (L0)", "## Relevant Memories (L0)")
L1_HEADING_PREFIX = "## Knowledge Graph Insights (L1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_dir(run_id: str) -> Path:
    return ARTIFACTS_DIR / f"personamem-{run_id}"


def _source_dir(run_id: str, source_eval_id: str) -> Path:
    return _run_dir(run_id) / "end_to_end" / source_eval_id


def _ablation_dir(run_id: str, ablation_id: str) -> Path:
    return _run_dir(run_id) / "official_context_ablation" / ablation_id


def _latest_completed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "completed" and row.get("question_id"):
            latest[row["question_id"]] = row
    return latest


def _require_source(run_id: str, source_eval_id: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    source_dir = _source_dir(run_id, source_eval_id)
    manifest_path = source_dir / "manifest.json"
    questions_path = source_dir / "question_suite.json"
    results_path = source_dir / SOURCE_RESULT_FILENAME
    missing = [path for path in (manifest_path, questions_path, results_path) if not path.is_file()]
    if missing:
        raise SystemExit("Missing primary-evaluation files:\n" + "\n".join(str(path) for path in missing))
    manifest = _read_json(manifest_path)
    questions = _read_json(questions_path)
    completed = _latest_completed(_read_jsonl(results_path))
    missing_ids = [question["question_id"] for question in questions if question["question_id"] not in completed]
    if missing_ids:
        raise SystemExit(f"Primary evaluation is incomplete; missing successful results: {missing_ids}")
    return source_dir, manifest, questions, completed


def _require_ablation(run_id: str, ablation_id: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    output_dir = _ablation_dir(run_id, ablation_id)
    manifest_path = output_dir / "manifest.json"
    slices_path = output_dir / SLICE_FILENAME
    if not manifest_path.is_file() or not slices_path.is_file():
        raise SystemExit(f"Prepared official-context ablation not found: {output_dir}\nRun prepare first.")
    return output_dir, _read_json(manifest_path), _read_jsonl(slices_path)


def _split_official_context(context: str) -> tuple[str, str]:
    """Split only at headings produced by RGMem's original formatter."""
    stripped = context.strip()
    if stripped == NO_MEMORY_CONTEXT:
        return NO_MEMORY_CONTEXT, NO_MEMORY_CONTEXT

    l1_position = stripped.find(L1_HEADING_PREFIX)
    if l1_position < 0:
        raise ValueError("Original context has no RGMem L1 heading; refusing to guess a split boundary.")

    l0_context = stripped[:l1_position].strip()
    l1_context = stripped[l1_position:].strip()
    if not l1_context.startswith(L1_HEADING_PREFIX):
        raise ValueError("Malformed RGMem L1 section.")
    if l0_context and not any(l0_context.startswith(heading) for heading in L0_HEADINGS):
        raise ValueError("Original context prefix is not an RGMem L0 section; refusing to truncate by character count.")
    if not l0_context:
        l0_context = NO_MEMORY_CONTEXT
    return l0_context, l1_context


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {options[letter]}" for letter in "ABCD")


def _answer_question_text(case: dict[str, Any]) -> str:
    return (
        f"{case['question']}\n\nOptions:\n{_format_options(case['options'])}\n\n"
        "Return exactly one capital letter: A, B, C, or D."
    )


def _parse_choice(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().upper()
    exact = re.fullmatch(r"[\(\[]?([A-D])[\)\]\.:]?", text)
    if exact:
        return exact.group(1)
    match = re.search(r"(?:ANSWER|CHOICE)\s*(?:IS|:)?\s*\(?([A-D])\)?", text)
    return match.group(1) if match else None


def _protocol_text() -> str:
    return """# RGMem official-context ablation protocol

1. Source data are completed rows from the original RGMem primary evaluation.
2. Query planning and retrieval are never rerun by this wrapper.
3. `l0_plus_l1` reuses the saved `retrieved_context` byte for byte.
4. `l0_only` and `l1_only` are structural slices at headings emitted by
   `l2_retrieval.formatter`; no keywords, entities, facts, or summaries are
   generated or rewritten by this wrapper.
5. `no_memory` uses RGMem formatter's standard empty-context string.
6. Every variant uses the same official answer-generation prompt and the same
   multiple-choice adapter. Gold answers are used only after the model call.
7. This wrapper performs no Elasticsearch, Neo4j, Redis, L0, L1, or L2 calls.
"""


def stage_prepare(args: argparse.Namespace) -> None:
    source_dir, source_manifest, questions, completed = _require_source(args.run_id, args.source_eval_id)
    output_dir = _ablation_dir(args.run_id, args.ablation_id)
    if output_dir.exists():
        raise SystemExit(f"Ablation already exists: {output_dir}\nUse a new --ablation-id.")

    slices = []
    for question in questions:
        qid = question["question_id"]
        source = completed[qid]
        original_context = source["retrieved_context"]
        l0_context, l1_context = _split_official_context(original_context)
        case = {
            "question_id": qid,
            "suite": question["suite"],
            "question": question["question"],
            "options": question["options"],
            "correct_answer": question["correct_answer"],
            "source_query_model": source.get("query_model"),
            "source_answer_model": source.get("answer_model"),
            "structured_query_raw": source.get("structured_query_raw"),
            "entity_queries": source.get("entity_queries"),
            "contexts": {
                "no_memory": NO_MEMORY_CONTEXT,
                "l0_only": l0_context,
                "l1_only": l1_context,
                "l0_plus_l1": original_context,
            },
            "context_chars": {
                "no_memory": len(NO_MEMORY_CONTEXT),
                "l0_only": len(l0_context),
                "l1_only": len(l1_context),
                "l0_plus_l1": len(original_context),
            },
            "context_sha256": {
                "no_memory": _sha256_text(NO_MEMORY_CONTEXT),
                "l0_only": _sha256_text(l0_context),
                "l1_only": _sha256_text(l1_context),
                "l0_plus_l1": _sha256_text(original_context),
            },
        }
        slices.append(case)

    # Create the artifact directory only after every saved official context has
    # passed the structural split checks. A malformed source therefore cannot
    # leave a half-prepared ablation that blocks a clean retry.
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / SLICE_FILENAME, slices)
    _write_text(output_dir / "protocol.md", _protocol_text())
    _write_json(output_dir / "manifest.json", {
        "ablation_id": args.ablation_id,
        "source_run_id": args.run_id,
        "source_eval_id": args.source_eval_id,
        "source_directory": str(source_dir),
        "created_at": _utc_now(),
        "question_count": len(slices),
        "variants": list(VARIANTS),
        "source_context_policy": "slice saved original RGMem retrieved_context by formatter headings",
        "retrieval_rerun": False,
        "database_access": False,
        "gold_sent_to_model": False,
        "source_primary_manifest": source_manifest,
    })
    print(f"Prepared official-context ablation: {output_dir}")
    print(f"Questions: {len(slices)}")
    print("No API calls or database connections were made.")


def _initialize_answer_pipeline() -> dict[str, Any]:
    from scripts.evaluate_l0_plus_l1 import ANSWER_GENERATION_PROMPT
    from config import settings
    from l0_memory.llm_interface.factory import LlmFactory

    provider = settings.LLM_PROVIDER
    cfg = settings.L0_LLM_CONFIGS.get(provider)
    if not cfg:
        raise ValueError(f"Missing L0 provider configuration: {provider}")
    answer_model = cfg["task_models"]["eval_chat_model"]
    client_config = {
        "model": answer_model,
        "api_key": cfg.get("api_key"),
        "base_url": cfg.get("base_url"),
        **cfg.get("params", {}),
    }
    answer_client = LlmFactory.create(
        provider,
        client_config,
        "RGMemOfficialContextAblationAnswerClient",
    )
    return {
        "answer_prompt": ANSWER_GENERATION_PROMPT,
        "answer_model": answer_model,
        "answer_client": answer_client,
    }


def stage_evaluate(args: argparse.Namespace) -> None:
    if not args.confirm_api:
        raise SystemExit("This calls the configured answer LLM. Re-run with --confirm-api.")
    output_dir, manifest, cases = _require_ablation(args.run_id, args.ablation_id)
    selected_cases = cases if args.suite == "all" else [case for case in cases if case["suite"] == args.suite]
    selected_variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    result_path = output_dir / RESULT_FILENAME
    completed_keys = {
        (row.get("question_id"), row.get("variant"))
        for row in _read_jsonl(result_path)
        if row.get("status") == "completed"
    }
    pipeline = _initialize_answer_pipeline()
    completed_now = skipped = 0

    for case in selected_cases:
        for variant in selected_variants:
            key = (case["question_id"], variant)
            if key in completed_keys:
                skipped += 1
                continue
            context = case["contexts"][variant]
            answer_prompt = pipeline["answer_prompt"].format(
                context=context,
                question=_answer_question_text(case),
            )
            trace_dir = output_dir / "traces" / case["question_id"] / variant
            _write_text(trace_dir / "01_context.md", context)
            _write_text(trace_dir / "02_exact_answer_prompt.txt", answer_prompt)
            base = {
                "question_id": case["question_id"],
                "suite": case["suite"],
                "variant": variant,
                "source_eval_id": manifest["source_eval_id"],
                "source_query_model": case.get("source_query_model"),
                "source_answer_model": case.get("source_answer_model"),
                "replay_answer_model": pipeline["answer_model"],
                "context_chars": len(context),
                "context_sha256": _sha256_text(context),
                "prompt_sha256": _sha256_text(answer_prompt),
                "retrieval_rerun": False,
                "gold_sent_to_model": False,
                "started_at": _utc_now(),
            }
            try:
                raw_value = pipeline["answer_client"].get_completion(
                    answer_prompt,
                    model_override=pipeline["answer_model"],
                )
                raw_answer = raw_value if isinstance(raw_value, str) else json.dumps(raw_value, ensure_ascii=False)
                _write_text(trace_dir / "03_model_raw_output.txt", raw_answer)
                choice = _parse_choice(raw_answer)
                if choice is None:
                    row = {
                        **base,
                        "status": "failed",
                        "failed_at": _utc_now(),
                        "error_type": "EMPTY_OR_UNPARSEABLE_ANSWER",
                        "raw_answer": raw_answer,
                    }
                else:
                    row = {
                        **base,
                        "status": "completed",
                        "completed_at": _utc_now(),
                        "question": case["question"],
                        "options": case["options"],
                        "correct_answer": case["correct_answer"],
                        "choice": choice,
                        "correct": choice == case["correct_answer"],
                        "raw_answer": raw_answer,
                    }
                    completed_now += 1
                    completed_keys.add(key)
            except Exception as exc:
                row = {
                    **base,
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            _write_json(trace_dir / "04_result.json", row)
            _append_jsonl(result_path, row)
            print(f"[{row['status']}] {case['question_id']} / {variant}")

    requested = len(selected_cases) * len(selected_variants)
    print(
        f"Official-context ablation finished: variants={','.join(selected_variants)}, "
        f"requested={requested}, completed_now={completed_now}, skipped_existing={skipped}"
    )
    print(f"Results: {result_path}")


def _latest_completed_pairs(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("question_id"), row.get("variant"))
        if row.get("status") == "completed" and key[0] and key[1] in VARIANTS:
            latest[key] = row
    return latest


def stage_status(args: argparse.Namespace) -> None:
    output_dir, manifest, cases = _require_ablation(args.run_id, args.ablation_id)
    completed = _latest_completed_pairs(_read_jsonl(output_dir / RESULT_FILENAME))
    ids = {case["question_id"] for case in cases}
    print(f"Ablation: {manifest['ablation_id']}")
    print(f"Source primary evaluation: {manifest['source_eval_id']}")
    for variant in VARIANTS:
        count = sum((qid, variant) in completed for qid in ids)
        print(f"{variant}: completed={count}/{len(ids)}")
    print(f"Directory: {output_dir}")


def stage_report(args: argparse.Namespace) -> None:
    output_dir, manifest, cases = _require_ablation(args.run_id, args.ablation_id)
    rows = _read_jsonl(output_dir / RESULT_FILENAME)
    completed = _latest_completed_pairs(rows)
    stats = defaultdict(lambda: {"completed": 0, "correct": 0})
    suite_stats = defaultdict(lambda: {"completed": 0, "correct": 0})
    for (qid, variant), row in completed.items():
        stats[variant]["completed"] += 1
        stats[variant]["correct"] += int(row["correct"])
        suite_stats[(row["suite"], variant)]["completed"] += 1
        suite_stats[(row["suite"], variant)]["correct"] += int(row["correct"])

    report = {
        "ablation_id": manifest["ablation_id"],
        "source_eval_id": manifest["source_eval_id"],
        "generated_at": _utc_now(),
        "overall": dict(stats),
        "by_suite": {f"{suite}/{variant}": value for (suite, variant), value in suite_stats.items()},
        "per_question": [],
        "unresolved_failures": [],
    }
    lines = [
        "# RGMem official-context ablation report", "",
        f"- Source primary evaluation: `{manifest['source_eval_id']}`",
        "- Retrieval rerun: no",
        "- Context source: structural slices of saved original RGMem context", "",
        "| Variant | Correct | Completed | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = stats[variant]
        accuracy = 100 * item["correct"] / item["completed"] if item["completed"] else 0.0
        lines.append(f"| {variant} | {item['correct']} | {item['completed']} | {accuracy:.2f}% |")

    lines += ["", "| Question | Suite | Gold | No memory | L0 | L1 | L0+L1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        qid = case["question_id"]
        outcomes = {}
        cells = []
        for variant in VARIANTS:
            row = completed.get((qid, variant))
            outcomes[variant] = {
                "choice": row.get("choice") if row else None,
                "correct": row.get("correct") if row else None,
            }
            cells.append(row["choice"] if row else "—")
        lines.append(f"| {qid} | {case['suite']} | {case['correct_answer']} | {' | '.join(cells)} |")
        report["per_question"].append({
            "question_id": qid,
            "suite": case["suite"],
            "correct_answer": case["correct_answer"],
            "outcomes": outcomes,
        })

    successful_keys = set(completed)
    report["unresolved_failures"] = [
        row for row in rows
        if row.get("status") == "failed"
        and (row.get("question_id"), row.get("variant")) not in successful_keys
    ]
    _write_json(output_dir / "official_context_ablation_report.json", report)
    _write_text(output_dir / "official_context_ablation_report.md", "\n".join(lines) + "\n")
    print(f"JSON report: {output_dir / 'official_context_ablation_report.json'}")
    print(f"Markdown report: {output_dir / 'official_context_ablation_report.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablate L0/L1 blocks from saved original RGMem primary contexts",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Slice saved original contexts; no API/database access")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--source-eval-id", required=True)
    prepare.add_argument("--ablation-id", required=True)
    prepare.set_defaults(func=stage_prepare)

    evaluate = commands.add_parser("evaluate", help="Replay selected context slices through the answer model")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--ablation-id", required=True)
    evaluate.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    evaluate.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    evaluate.add_argument("--confirm-api", action="store_true")
    evaluate.set_defaults(func=stage_evaluate)

    status = commands.add_parser("status", help="Show replay completion counts")
    status.add_argument("--run-id", required=True)
    status.add_argument("--ablation-id", required=True)
    status.set_defaults(func=stage_status)

    report = commands.add_parser("report", help="Build local replay reports")
    report.add_argument("--run-id", required=True)
    report.add_argument("--ablation-id", required=True)
    report.set_defaults(func=stage_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
