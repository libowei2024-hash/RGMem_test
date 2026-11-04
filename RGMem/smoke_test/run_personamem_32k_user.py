"""Generic one-user PersonaMem-32K runner built on the proven smoke-test engine.

This file is only an orchestration shell. It does not implement L0, L1, graph
updates, inference, consolidation, or retrieval. Those operations are delegated
to ``run_personamem_single_user.py``, which in turn calls RGMem's native code.

Dataset users are addressed with a one-based ``--user-number``. To preserve the
protocol of the successful Kanoa run, the shell deterministically selects the
first shared trajectory listed for that persona in ``questions_32k.csv``. The
selected context id remains recorded in the run manifest for provenance, but it
is not a parameter the experiment operator needs to manage.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


SMOKE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SMOKE_ROOT.parent
DATA_ROOT = PROJECT_ROOT.parent / "data"
DEFAULT_QUESTIONS = DATA_ROOT / "questions_32k.csv"
DEFAULT_CONTEXTS = DATA_ROOT / "shared_contexts_32k.jsonl"
PROBE_ROOT = SMOKE_ROOT / "probes" / "32k"
GENERIC_PROBE_QUERY = (
    "What are the user's current preferences, preference changes, stable interests, "
    "explicit dislikes, and important conditions or exceptions?"
)

if str(SMOKE_ROOT) not in sys.path:
    sys.path.insert(0, str(SMOKE_ROOT))

import run_personamem_single_user as engine


def _read_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_contexts(path: Path) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
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


def _persona_name(messages: list[dict[str, Any]], persona_id: int) -> str:
    system_text = next(
        (str(item.get("content", "")) for item in messages if item.get("role") == "system"),
        "",
    )
    name_match = re.search(r"\bName:\s*([^\n]+)", system_text)
    if name_match:
        return name_match.group(1).strip()
    meet_match = re.search(r"\bMeet\s+([^,\n]+)", system_text)
    if meet_match:
        return meet_match.group(1).strip()
    return f"Persona {persona_id}"


def _registry(questions_path: Path, contexts_path: Path) -> list[dict[str, Any]]:
    questions = _read_questions(questions_path)
    contexts = _read_contexts(contexts_path)
    persona_ids = sorted({int(row["persona_id"]) for row in questions})
    registry = []
    for user_number, persona_id in enumerate(persona_ids, 1):
        persona_questions = [row for row in questions if int(row["persona_id"]) == persona_id]
        context_ids: list[str] = []
        for row in persona_questions:
            context_id = row["shared_context_id"]
            if context_id not in context_ids:
                context_ids.append(context_id)
        if not context_ids:
            raise ValueError(f"Persona {persona_id} has no shared context")
        missing = [context_id for context_id in context_ids if context_id not in contexts]
        if missing:
            raise ValueError(f"Persona {persona_id} has missing contexts: {missing}")
        primary_context_id = context_ids[0]
        primary_messages = contexts[primary_context_id]
        registry.append(
            {
                "dataset_user_number": user_number,
                "persona_id": persona_id,
                "persona_name": _persona_name(primary_messages, persona_id),
                "primary_context_id": primary_context_id,
                "available_context_ids": context_ids,
                "primary_message_count": len(primary_messages),
                "primary_question_count": sum(
                    row["shared_context_id"] == primary_context_id for row in persona_questions
                ),
            }
        )
    return registry


def _selection(
    user_number: int,
    questions_path: Path,
    contexts_path: Path,
) -> dict[str, Any]:
    registry = _registry(questions_path, contexts_path)
    if user_number < 1 or user_number > len(registry):
        raise SystemExit(f"--user-number must be between 1 and {len(registry)}")
    selected = dict(registry[user_number - 1])
    contexts = _read_contexts(contexts_path)
    selected["messages"] = contexts[selected["primary_context_id"]]
    selected["rounds"] = _source_sessions(selected["messages"])
    return selected


def _source_sessions(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "system"
    ]
    if not system_indices:
        raise ValueError("Selected PersonaMem context has no system/session markers")
    if system_indices[0] != 0:
        raise ValueError(f"First system/session marker is at {system_indices[0]}, expected 0")
    rounds = []
    for offset, start in enumerate(system_indices):
        end = system_indices[offset + 1] - 1 if offset + 1 < len(system_indices) else len(messages) - 1
        rounds.append(
            {
                "number": offset + 1,
                "start": start,
                "end": end,
                "label": f"source session {offset + 1}",
            }
        )
    return rounds


def _probe_path(user_number: int) -> Path:
    return PROBE_ROOT / f"user_{user_number:03d}.json"


def _empty_probe(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe_version": 1,
        "dataset": "PersonaMem-32K",
        "dataset_user_number": selection["dataset_user_number"],
        "persona_id": selection["persona_id"],
        "persona_name": selection["persona_name"],
        "status": "awaiting_user_definition",
        "gold_labels_active": False,
        "generic_query": GENERIC_PROBE_QUERY,
        "instructions": [
            "Add only probes grounded in this selected user's original messages.",
            "Review A and K labels before activating them for judgment.",
            "This file is snapshotted into each run and is never sent to L0 or L1 writes.",
        ],
        "events": [],
    }


def _load_probe(selection: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _probe_path(selection["dataset_user_number"])
    if not path.is_file():
        return path, _empty_probe(selection)
    probe = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_user_number": selection["dataset_user_number"],
        "persona_id": selection["persona_id"],
    }
    for key, value in expected.items():
        if probe.get(key) != value:
            raise ValueError(f"Probe file {path} has {key}={probe.get(key)!r}, expected {value!r}")
    return path, probe


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _dynamic_source_bundle(
    selection: dict[str, Any],
    questions_path: Path,
    contexts_path: Path,
) -> dict[str, Any]:
    contexts = _read_contexts(contexts_path)
    questions = _read_questions(questions_path)
    context_id = selection["primary_context_id"]
    messages = contexts[context_id]
    selected_questions = [
        row
        for row in questions
        if int(row["persona_id"]) == selection["persona_id"]
        and row["shared_context_id"] == context_id
    ]
    if not selected_questions:
        raise ValueError(f"No questions found for persona/context {selection['persona_id']}/{context_id}")
    if any(int(row["end_index_in_shared_context"]) >= len(messages) for row in selected_questions):
        raise ValueError("A selected question has an out-of-range history boundary")

    system_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "system"
    ]
    unexpected_roles = [
        (index, message.get("role"))
        for index, message in enumerate(messages)
        if message.get("role") not in {"system", "user", "assistant"}
    ]
    embedded_role_warnings = []
    for index, message in enumerate(messages):
        content = str(message.get("content", ""))
        role = message.get("role")
        if role == "assistant" and re.search(r"(?<!^)User:\s", content):
            embedded_role_warnings.append(
                {"index": index, "role": role, "warning": "assistant_contains_embedded_User"}
            )
        if role == "user" and re.search(r"(?<!^)Assistant:\s", content):
            embedded_role_warnings.append(
                {"index": index, "role": role, "warning": "user_contains_embedded_Assistant"}
            )
    return {
        "messages": messages,
        "questions": selected_questions,
        "system_indices": system_indices,
        "unexpected_roles": unexpected_roles,
        "embedded_role_warnings": embedded_role_warnings,
    }


def _configure_engine(
    selection: dict[str, Any],
    questions_path: Path,
    contexts_path: Path,
    probe_source: tuple[Path, dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    probe_path, probe = probe_source or _load_probe(selection)
    engine.PERSONA_ID = selection["persona_id"]
    engine.PERSONA_NAME = selection["persona_name"]
    engine.CONTEXT_ID = selection["primary_context_id"]
    engine.ROUNDS = copy.deepcopy(selection["rounds"])
    engine.PROBE_QUERY = probe.get("generic_query") or GENERIC_PROBE_QUERY
    engine._diagnostic_label_template = lambda: copy.deepcopy(probe)
    engine._source_bundle = lambda q, c: _dynamic_source_bundle(selection, q, c)
    return probe_path, probe


def _run_id(user_number: int, attempt: int) -> str:
    if attempt < 1 or attempt > 99:
        raise SystemExit("--attempt must be between 1 and 99")
    return f"p32k-u{user_number:02d}-a{attempt:02d}"


def _augment_manifest(
    run_id: str,
    selection: dict[str, Any],
    probe_path: Path,
    probe: dict[str, Any],
    questions_path: Path,
    contexts_path: Path,
) -> None:
    manifest_path = engine._manifest_path(run_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "wrapper_protocol": "personamem-32k-one-user-v1",
            "dataset_user_number": selection["dataset_user_number"],
            "context_selection_policy": "first context listed for persona in questions_32k.csv",
            "available_context_count": len(selection["available_context_ids"]),
            "selected_context_ordinal": 1,
            "source_session_count": len(selection["rounds"]),
            "probe_registry_path": str(probe_path),
            "probe_registry_sha256": _sha256(probe_path),
            "probe_status_at_prepare": probe.get("status", "unknown"),
            "questions_sha256": _sha256(questions_path),
            "contexts_sha256": _sha256(contexts_path),
        }
    )
    engine._write_json(manifest_path, manifest)


def _prepare(
    selection: dict[str, Any],
    run_id: str,
    questions_path: Path,
    contexts_path: Path,
) -> None:
    probe_path, probe = _configure_engine(selection, questions_path, contexts_path)
    engine._prepare_artifacts(run_id, questions_path, contexts_path)
    _augment_manifest(run_id, selection, probe_path, probe, questions_path, contexts_path)
    print(f"Prepared user {selection['dataset_user_number']}: {selection['persona_name']}")
    print(f"Run: {run_id}")
    print(f"Source sessions: {len(selection['rounds'])}")
    print(f"Probe registry: {probe_path} [{probe.get('status', 'unknown')}]")
    print("No API or database writes were performed.")


def _assert_run_matches(run_id: str, selection: dict[str, Any]) -> None:
    manifest = engine._load_manifest(run_id)
    expected = {
        "persona_id": selection["persona_id"],
        "shared_context_id": selection["primary_context_id"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(
                f"Existing run {run_id} has {key}={manifest.get(key)!r}, expected {value!r}"
            )


def stage_list_users(args: argparse.Namespace) -> None:
    for item in _registry(args.questions, args.contexts):
        print(
            f"{item['dataset_user_number']:>2}: persona_id={item['persona_id']}; "
            f"name={item['persona_name']}; contexts={len(item['available_context_ids'])}; "
            f"selected_messages={item['primary_message_count']}; "
            f"selected_questions={item['primary_question_count']}"
        )


def stage_inspect(args: argparse.Namespace) -> None:
    selected = _selection(args.user_number, args.questions, args.contexts)
    probe_path, probe = _load_probe(selected)
    print(f"Dataset user: {selected['dataset_user_number']}")
    print(f"Persona: {selected['persona_id']} / {selected['persona_name']}")
    print(f"Messages: {selected['primary_message_count']}")
    print(f"Questions: {selected['primary_question_count']}")
    print(f"Source sessions: {len(selected['rounds'])}")
    for item in selected["rounds"]:
        print(f"  {item['number']}: {item['start']}-{item['end']}")
    print(f"Probe registry: {probe_path} [{probe.get('status', 'unknown')}]")


def stage_prepare(args: argparse.Namespace) -> None:
    selected = _selection(args.user_number, args.questions, args.contexts)
    run_id = _run_id(args.user_number, args.attempt)
    _prepare(selected, run_id, args.questions, args.contexts)


def _has_probe_events(probe: dict[str, Any]) -> bool:
    return bool(probe.get("events"))


def stage_run_all(args: argparse.Namespace) -> None:
    if not args.confirm_api_and_write:
        raise SystemExit("This runs API/database stages. Re-run with --confirm-api-and-write.")
    selected = _selection(args.user_number, args.questions, args.contexts)
    run_id = _run_id(args.user_number, args.attempt)
    run_dir = engine._run_directory(run_id)
    if not run_dir.exists():
        _prepare(selected, run_id, args.questions, args.contexts)
    else:
        run_probe_path = run_dir / "diagnostic_labels.json"
        if not run_probe_path.is_file():
            raise SystemExit(f"Run probe snapshot is missing: {run_probe_path}")
        run_probe = json.loads(run_probe_path.read_text(encoding="utf-8"))
        _configure_engine(
            selected,
            args.questions,
            args.contexts,
            probe_source=(run_probe_path, run_probe),
        )
        _assert_run_matches(run_id, selected)
        print(f"Resuming existing run without replaying completed stages: {run_id}")

    # Freeze probe semantics per run. A later edit to the registry must not
    # silently change a partially completed experiment during resume.
    run_probe_path = run_dir / "diagnostic_labels.json"
    if not run_probe_path.is_file():
        raise SystemExit(f"Run probe snapshot is missing: {run_probe_path}")
    probe = json.loads(run_probe_path.read_text(encoding="utf-8"))
    probe_path = run_probe_path
    engine.PROBE_QUERY = probe.get("generic_query") or GENERIC_PROBE_QUERY

    if args.with_probes and not _has_probe_events(probe):
        raise SystemExit(
            f"--with-probes was requested, but {probe_path} has no probe events. "
            "Define and review the user-specific probes first."
        )

    engine.stage_preflight(argparse.Namespace())
    for spec in selected["rounds"]:
        number = spec["number"]
        round_dir = engine._round_directory(run_id, number)
        if not (round_dir / ".completed").exists():
            if (round_dir / ".started").exists():
                raise SystemExit(
                    f"Round {number} has an interrupted/partial start: {round_dir}. "
                    "Evidence was preserved; no automatic replay was attempted."
                )
            engine.stage_round(
                argparse.Namespace(
                    run_id=run_id,
                    round=number,
                    confirm_api_and_write=True,
                    with_diagnostic_probes=args.with_probes,
                )
            )
        else:
            print(f"[skip completed] round {number}")

        consolidation_dir = round_dir / "consolidation"
        if not (consolidation_dir / ".completed").exists():
            if (consolidation_dir / ".started").exists():
                raise SystemExit(
                    f"Consolidation after round {number} has an interrupted/partial start: "
                    f"{consolidation_dir}. Evidence was preserved; no automatic replay was attempted."
                )
            engine.stage_consolidate(
                argparse.Namespace(
                    run_id=run_id,
                    after_round=number,
                    confirm_api_and_write=True,
                    with_diagnostic_probes=args.with_probes,
                )
            )
        else:
            print(f"[skip completed] consolidation {number}")

    engine.stage_status(argparse.Namespace(run_id=run_id))
    engine.stage_report(argparse.Namespace(run_id=run_id))


def stage_status(args: argparse.Namespace) -> None:
    selected = _selection(args.user_number, args.questions, args.contexts)
    run_id = _run_id(args.user_number, args.attempt)
    run_probe_path = engine._run_directory(run_id) / "diagnostic_labels.json"
    if not run_probe_path.is_file():
        raise SystemExit(f"Run probe snapshot is missing: {run_probe_path}")
    run_probe = json.loads(run_probe_path.read_text(encoding="utf-8"))
    _configure_engine(
        selected,
        args.questions,
        args.contexts,
        probe_source=(run_probe_path, run_probe),
    )
    _assert_run_matches(run_id, selected)
    engine.stage_status(argparse.Namespace(run_id=run_id))


def stage_report(args: argparse.Namespace) -> None:
    selected = _selection(args.user_number, args.questions, args.contexts)
    run_id = _run_id(args.user_number, args.attempt)
    run_probe_path = engine._run_directory(run_id) / "diagnostic_labels.json"
    if not run_probe_path.is_file():
        raise SystemExit(f"Run probe snapshot is missing: {run_probe_path}")
    run_probe = json.loads(run_probe_path.read_text(encoding="utf-8"))
    _configure_engine(
        selected,
        args.questions,
        args.contexts,
        probe_source=(run_probe_path, run_probe),
    )
    _assert_run_matches(run_id, selected)
    engine.stage_report(argparse.Namespace(run_id=run_id))


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)


def _add_run_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--user-number", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    _add_source_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-user-at-a-time PersonaMem-32K RGMem orchestration shell"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_users = commands.add_parser("list-users", help="List stable one-based dataset users")
    _add_source_args(list_users)
    list_users.set_defaults(handler=stage_list_users)

    inspect = commands.add_parser("inspect", help="Inspect one user; no API/database access")
    inspect.add_argument("--user-number", type=int, required=True)
    _add_source_args(inspect)
    inspect.set_defaults(handler=stage_inspect)

    prepare = commands.add_parser("prepare", help="Prepare one isolated user run")
    _add_run_selector(prepare)
    prepare.set_defaults(handler=stage_prepare)

    run_all = commands.add_parser("run-all", help="Prepare/resume and run every source session")
    _add_run_selector(run_all)
    run_all.add_argument("--with-probes", action="store_true")
    run_all.add_argument("--confirm-api-and-write", action="store_true")
    run_all.set_defaults(handler=stage_run_all)

    status = commands.add_parser("status", help="Show local status for one isolated run")
    _add_run_selector(status)
    status.set_defaults(handler=stage_status)

    report = commands.add_parser("report", help="Aggregate ADD/UPDATE audit for one run")
    _add_run_selector(report)
    report.set_defaults(handler=stage_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
