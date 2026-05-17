from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


CONTEXT_PACKS: dict[str, dict[str, Any]] = {
    "repo-fast": {
        "pack": "agent_context/repo_fast_pack.md",
        "required_read_files": ["agent_protocols/AGENT_BRIEF.md", "agent_context/repo_fast_pack.md"],
        "allowed_edit_scope": [],
        "forbidden_edit_scope": ["implementation files unless a failing check is explicitly in scope"],
        "validation_commands": ["python scripts/repo_doctor.py fast"],
        "context_budget": "brief + fast pack + command output only",
        "escalation_conditions": ["fast gate fails", "state-changing request appears"],
    },
    "route-config": {
        "pack": "agent_context/route_config_pack.md",
        "required_read_files": [
            "agent_protocols/AGENT_BRIEF.md",
            "agent_context/route_config_pack.md",
            "configs/statuses.yaml",
            "src/hust_bci_er/config/schema.py",
        ],
        "allowed_edit_scope": ["configs/routes/models/", "tests/test_route_schema.py"],
        "forbidden_edit_scope": ["datasets", "splits", "route status without audit evidence", "existing route results"],
        "validation_commands": ["python scripts/validate_route.py --all", "python scripts/repo_doctor.py fast"],
        "context_budget": "route config pack plus changed route and schema/test references",
        "escalation_conditions": ["status promotion requested", "split/evaluation protocol change required"],
    },
    "model-backbone": {
        "pack": "agent_context/model_backbone_pack.md",
        "required_read_files": [
            "agent_protocols/AGENT_BRIEF.md",
            "agent_context/model_backbone_pack.md",
            "src/hust_bci_er/models/factory.py",
            "scripts/model_smoke.py",
            "tests/test_backbone_guards.py",
        ],
        "allowed_edit_scope": ["src/hust_bci_er/models/", "src/hust_bci_er/training/", "tests/test_backbone_guards.py", "tests/test_real_adapter_new_models.py"],
        "forbidden_edit_scope": ["route summaries", "candidate evidence", "promotion evidence", "dataset/split definitions"],
        "validation_commands": ["python scripts/model_smoke.py", "python scripts/repo_doctor.py fast"],
        "context_budget": "model factory/backbone files and smoke tests only",
        "escalation_conditions": ["provenance/audit failure points outside model code", "route status change requested"],
    },
    "evidence-gate": {
        "pack": "agent_context/evidence_gate_pack.md",
        "required_read_files": [
            "agent_protocols/AGENT_BRIEF.md",
            "agent_context/evidence_gate_pack.md",
            "agent_protocols/experiment_audit.md",
            "scripts/audit_experiment.py",
            "src/hust_bci_er/audit/",
            "tests/test_audit_gate_behavior.py",
        ],
        "allowed_edit_scope": ["scripts/audit_experiment.py", "scripts/repo_doctor.py", "src/hust_bci_er/audit/", "tests/test_audit_gate_behavior.py"],
        "forbidden_edit_scope": ["model backbones", "route results", "candidate evidence artifacts"],
        "validation_commands": ["python -m pytest tests/test_audit_gate_behavior.py -q", "python scripts/repo_doctor.py fast"],
        "context_budget": "audit/schema/manifest tests and directly failing gate code",
        "escalation_conditions": ["requires changing evaluation metric semantics", "requires route status update"],
    },
    "protocol-runner": {
        "pack": "agent_context/protocol_runner_pack.md",
        "required_read_files": [
            "agent_protocols/AGENT_BRIEF.md",
            "agent_context/protocol_runner_pack.md",
            "docs/04_evaluation_protocols.md",
            "src/hust_bci_er/evaluation/protocols/runner.py",
            "src/hust_bci_er/evaluation/protocols/subject_splits.py",
            "scripts/run_evaluation_protocol.py",
            "scripts/run_route_job.py",
            "scripts/run_artifact_job.py",
            "tests/test_protocol_runner.py",
        ],
        "allowed_edit_scope": ["src/hust_bci_er/evaluation/protocols/", "scripts/run_evaluation_protocol.py", "scripts/run_route_job.py", "scripts/run_artifact_job.py", "tests/test_protocol_runner.py"],
        "forbidden_edit_scope": ["split definitions unless explicitly requested and reviewed", "route status", "model internals unless adapter tests fail"],
        "validation_commands": ["python -m pytest tests/test_protocol_runner.py -q", "python scripts/repo_doctor.py fast"],
        "context_budget": "protocol runner files plus targeted tests",
        "escalation_conditions": ["candidate-grade evidence change", "evaluation protocol semantics change"],
    },
    "score-fusion": {
        "pack": "agent_context/score_fusion_pack.md",
        "required_read_files": [
            "agent_protocols/AGENT_BRIEF.md",
            "agent_context/score_fusion_pack.md",
            "src/hust_bci_er/config/component_map.py",
            "src/hust_bci_er/inference/clean_score_routes.py",
            "src/hust_bci_er/inference/score_route_assembly.py",
            "scripts/run_score_fusion_routes.py",
            "scripts/export_component_scores.py",
            "tests/test_score_fusion_runner.py",
            "tests/test_score_route_assembly.py",
        ],
        "allowed_edit_scope": ["src/hust_bci_er/config/component_map.py", "src/hust_bci_er/inference/", "scripts/run_score_fusion_routes.py", "scripts/export_component_scores.py", "tests/test_score_fusion_runner.py", "tests/test_score_route_assembly.py"],
        "forbidden_edit_scope": ["unrelated model backbone internals", "datasets", "split definitions", "route statuses"],
        "validation_commands": ["python -m pytest tests/test_score_fusion_runner.py tests/test_score_route_assembly.py -q", "python scripts/repo_doctor.py fast"],
        "context_budget": "score-fusion pack, runner, component-map, score assembly tests",
        "escalation_conditions": ["validation failure points to real adapter/model internals", "component evidence is missing or untrusted"],
    },
    "summary-report": {
        "pack": "agent_context/summary_report_pack.md",
        "required_read_files": [
            "agent_protocols/AGENT_BRIEF.md",
            "agent_context/summary_report_pack.md",
            "src/hust_bci_er/evaluation/protocols/summary.py",
            "scripts/summarize_protocol_run.py",
            "scripts/check_summary_consistency.py",
            "tests/test_summary_consistency.py",
            "tests/test_protocol_runner.py",
        ],
        "allowed_edit_scope": ["src/hust_bci_er/evaluation/protocols/summary.py", "scripts/summarize_protocol_run.py", "scripts/check_summary_consistency.py", "tests/test_summary_consistency.py", "tests/test_protocol_runner.py"],
        "forbidden_edit_scope": ["metric values", "route status", "candidate/promotion evidence content"],
        "validation_commands": ["python -m pytest tests/test_summary_consistency.py tests/test_protocol_runner.py -q", "python scripts/repo_doctor.py fast"],
        "context_budget": "summary/report pack and summary consistency code",
        "escalation_conditions": ["summary would imply route promotion", "manifest/audit schema must change"],
    },
    "review": {
        "pack": "agent_context/review_pack.md",
        "required_read_files": ["agent_protocols/AGENT_BRIEF.md", "agent_context/review_pack.md"],
        "allowed_edit_scope": [],
        "forbidden_edit_scope": ["implementation edits", "PR comment resolution"],
        "validation_commands": [],
        "context_budget": "diff, review pack, and directly referenced files",
        "escalation_conditions": ["review request turns into fix request", "state-changing action requested"],
    },
    "review-fix": {
        "pack": "agent_context/review_fix_pack.md",
        "required_read_files": [
            "agent_protocols/AGENT_BRIEF.md",
            "agent_context/review_fix_pack.md",
            "scripts/agent_review_inbox.py",
            "scripts/agent_plan_ingest.py",
            "scripts/agent_session.py",
        ],
        "allowed_edit_scope": ["files referenced by the issue board", "tests covering fixed issues"],
        "forbidden_edit_scope": ["unrelated files", "route status/evidence changes unless the issue board explicitly requires and audit supports them"],
        "validation_commands": ["targeted tests for fixed issues", "python scripts/repo_doctor.py fast"],
        "context_budget": "review-fix pack, issue board, selected files only",
        "escalation_conditions": ["review comments conflict", "issue requires state-changing evidence update"],
    },
    "context-engineering": {
        "pack": "agent_context/context_engineering_pack.md",
        "required_read_files": [
            "agent_protocols/AGENT_BRIEF.md",
            "agent_context/context_engineering_pack.md",
            "AGENTS.md",
            "agent_protocols/skill_router.md",
            "docs/目录怎么用.md",
            "tests/test_agent_protocols.py",
        ],
        "allowed_edit_scope": ["AGENTS.md", "agent_protocols/", "agent_context/", "scripts/agent_*.py", "tests/test_agent_intake_workflow.py", "tests/test_agent_protocols.py", "docs/目录怎么用.md"],
        "forbidden_edit_scope": ["model behavior", "route status", "experiment metrics", "datasets", "split definitions", "candidate evidence", "promotion evidence", "existing route results"],
        "validation_commands": [
            "python -m pytest tests/test_agent_intake_workflow.py tests/test_agent_protocols.py -q",
            "python scripts/agent_intake.py --message \"score-fusion：修 component map 的 whitening 绑定问题\"",
            "python scripts/agent_context.py --task score-fusion",
            "python scripts/repo_doctor.py fast",
        ],
        "context_budget": "brief, context-engineering pack, router docs, agent scripts/tests",
        "escalation_conditions": ["request expands into route/evidence changes", "intake classifier needs non-deterministic behavior"],
    },
}


ALIASES = {
    "repo": "repo-fast",
    "fast": "repo-fast",
    "score_fusion": "score-fusion",
    "review_fix": "review-fix",
    "model_backbone": "model-backbone",
    "protocol_runner": "protocol-runner",
    "summary_report": "summary-report",
    "evidence_gate": "evidence-gate",
}


def normalize_task(task: str) -> str:
    key = task.strip().lower().replace("_", "-")
    key = ALIASES.get(key, key)
    if key not in CONTEXT_PACKS:
        valid = ", ".join(sorted(CONTEXT_PACKS))
        raise ValueError(f"unknown task family: {task}; valid: {valid}")
    return key


def context_for_task(task: str) -> dict[str, Any]:
    key = normalize_task(task)
    data = dict(CONTEXT_PACKS[key])
    data["task_family"] = key
    data["context_pack_path"] = data.pop("pack")
    return data


def print_human(data: dict[str, Any]) -> None:
    print(f"Task family: {data['task_family']}")
    print(f"Context pack: {data['context_pack_path']}")
    for key in [
        "required_read_files",
        "allowed_edit_scope",
        "forbidden_edit_scope",
        "validation_commands",
        "escalation_conditions",
    ]:
        print(f"{key}:")
        values = data.get(key) or []
        if not values:
            print("  - none")
        for value in values:
            print(f"  - {value}")
    print(f"context_budget: {data['context_budget']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select minimal context pack for a repository task family.")
    parser.add_argument("--list", action="store_true", help="List supported task families.")
    parser.add_argument("--task", help="Task family to select.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    args = parser.parse_args(argv)

    if args.list:
        for task in sorted(CONTEXT_PACKS):
            print(task)
        return 0
    if not args.task:
        parser.error("--task is required unless --list is used")
    try:
        data = context_for_task(args.task)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print_human(data)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print_human(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
