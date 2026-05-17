from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.agent_context import CONTEXT_PACKS, context_for_task
from scripts.agent_intake import classify, classify_with_deepseek, main as intake_main, maybe_refine_with_deepseek
from scripts.agent_plan_ingest import ingest, parse_text
from scripts.agent_review_inbox import (
    classify_pr_comment,
    infer_review_severity,
    ingest_review_inbox,
    issues_from_payload,
    looks_like_copilot_digest,
    parse_copilot_digest,
)
from scripts.agent_session import init_session


ROOT = Path(__file__).resolve().parents[1]
TASK_FAMILIES = {
    "repo-fast",
    "route-config",
    "model-backbone",
    "evidence-gate",
    "protocol-runner",
    "score-fusion",
    "summary-report",
    "review",
    "review-fix",
    "context-engineering",
}
PACKS = {
    "repo_fast_pack.md",
    "route_config_pack.md",
    "model_backbone_pack.md",
    "evidence_gate_pack.md",
    "protocol_runner_pack.md",
    "score_fusion_pack.md",
    "summary_report_pack.md",
    "review_pack.md",
    "review_fix_pack.md",
    "context_engineering_pack.md",
}


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_all_context_pack_files_exist():
    for pack in PACKS:
        assert (ROOT / "agent_context" / pack).exists(), pack


def test_agent_context_list_includes_all_task_families():
    proc = run_script("scripts/agent_context.py", "--list")
    observed = set(proc.stdout.split())
    assert TASK_FAMILIES <= observed


def test_agent_context_review_fix_includes_issue_board_and_session_workflow():
    proc = run_script("scripts/agent_context.py", "--task", "review-fix")
    assert "agent_context/review_fix_pack.md" in proc.stdout
    assert "scripts/agent_plan_ingest.py" in proc.stdout
    assert "scripts/agent_session.py" in proc.stdout


def test_agent_context_score_fusion_is_narrow():
    proc = run_script("scripts/agent_context.py", "--task", "score-fusion")
    assert "scripts/run_score_fusion_routes.py" in proc.stdout
    assert "src/hust_bci_er/config/component_map.py" in proc.stdout
    assert "src/hust_bci_er/models/backbones" not in proc.stdout


def test_long_s0_s1_review_plan_is_plan_driven():
    message = """
S0-1 Blocking score fusion artifact gap
scripts/run_score_fusion_routes.py must write manifest.json.

S0-2 No-Go protocol evidence gap
src/hust_bci_er/evaluation/protocols/runner.py needs failed_jobs accounting.

Validation:
python -m pytest tests/test_score_fusion_runner.py -q
python scripts/repo_doctor.py fast
"""
    result = classify(message)
    assert result["mode"] == "plan-driven"
    assert result["needs_plan_ingest"] is True
    assert result["should_edit"] is True


def test_pr_review_fix_intake():
    result = classify("修 PR17 unresolved review")
    assert result["mode"] == "review-fix"
    assert result["needs_pr_review_fetch"] is True
    assert result["should_edit"] is True


def test_read_only_pr_review_intake_does_not_allow_edits():
    for message in ["review PR18", "深度review PR18", "看一下 PR18", "看一下 PR18 conversation review", "review PR18 comments"]:
        result = classify(message)
        assert result["mode"] == "planning-only"
        assert result["task_family"] == "review"
        assert result["should_edit"] is False


def test_unresolved_pr_review_fix_intake_allows_edits():
    for message in ["修 PR18 unresolved review", "修 PR18 conversation review", "处理 PR18 comments"]:
        result = classify(message)
        assert result["mode"] == "review-fix"
        assert result["task_family"] == "review-fix"
        assert result["should_edit"] is True


def test_evaluation_protocol_planning_is_not_state_changing():
    for message in ["which evaluation protocol should I use?", "帮我分析 P1/P2/P3 评估协议"]:
        result = classify(message)
        assert result["mode"] == "planning-only"
        assert result["should_edit"] is False


def test_protected_protocol_and_split_changes_are_state_changing():
    for message in ["修改 evaluation protocol", "删除 split evidence"]:
        result = classify(message)
        assert result["mode"] == "state-changing"
        assert result["should_ask_human"] is True


def test_score_fusion_direct_task_intake():
    result = classify("score-fusion：修 component map 的 whitening 绑定问题")
    assert result["mode"] == "direct-task"
    assert result["task_family"] == "score-fusion"


def test_planning_only_intake():
    result = classify("现在这个仓库后面该怎么做")
    assert result["mode"] == "planning-only"
    assert result["should_edit"] is False


def test_command_only_intake():
    result = classify("跑 repo fast gate")
    assert result["mode"] == "command-only"


def test_state_changing_without_audit_asks_human():
    result = classify("把 route 改成 PROMOTED")
    assert result["mode"] == "state-changing"
    assert result["should_ask_human"] is True


def test_agent_plan_ingest_extracts_multiple_issues(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(
        """
### S0-1 score fusion manifest missing
Problem in scripts/run_score_fusion_routes.py.
需要补 manifest.json and score_matrix.csv.
python -m pytest tests/test_score_fusion_runner.py -q

### S0-2 protocol summary missing artifact jobs
Problem in src/hust_bci_er/evaluation/protocols/summary.py.
需要补 artifact-only expected artifact checks.
python -m pytest tests/test_protocol_runner.py -q
""",
        encoding="utf-8",
    )
    issues = ingest(plan, tmp_path / "session")
    board = json.loads((tmp_path / "session" / "issue_board.json").read_text(encoding="utf-8"))
    assert len(issues) >= 2
    assert len(board["issues"]) >= 2
    assert {issue["id"] for issue in board["issues"]} >= {"S0-1", "S0-2"}
    assert any("scripts/run_score_fusion_routes.py" in issue["files_hint"] for issue in board["issues"])


def test_agent_session_init_and_status(tmp_path):
    session_dir = init_session("context-engineering", tmp_path / "session", intake_mode="direct-task")
    assert (session_dir / "session.json").exists()
    assert (session_dir / "session.md").exists()
    proc = run_script("scripts/agent_session.py", "status", "--session-dir", str(session_dir))
    assert "context-engineering" in proc.stdout


def test_agent_session_tracks_reads_commands_validation_and_next_action(tmp_path):
    session_dir = init_session("context-engineering", tmp_path / "session", intake_mode="direct-task")
    run_script("scripts/agent_session.py", "read", "--session-dir", str(session_dir), "--file", "scripts/agent_intake.py")
    run_script("scripts/agent_session.py", "command", "--session-dir", str(session_dir), "--command", "python -m pytest tests/test_agent_intake_workflow.py -q")
    run_script("scripts/agent_session.py", "validation", "--session-dir", str(session_dir), "--status", "pass", "--message", "targeted tests passed")
    run_script("scripts/agent_session.py", "next", "--session-dir", str(session_dir), "--message", "run repo_doctor fast")
    data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    markdown = (session_dir / "session.md").read_text(encoding="utf-8")
    assert data["files_read"] == ["scripts/agent_intake.py"]
    assert data["commands_run"][0]["command"] == "python -m pytest tests/test_agent_intake_workflow.py -q"
    assert data["validation_status"] == "pass"
    assert data["next_action"] == "run repo_doctor fast"
    assert "scripts/agent_intake.py" in markdown
    assert "python -m pytest tests/test_agent_intake_workflow.py -q" in markdown


def test_agents_md_no_longer_requires_all_eight_full_docs_for_every_task():
    text = read("AGENTS.md")
    required = text.split("## Required Intake", 1)[1].split("## Skill Routing", 1)[0]
    assert "agent_protocols/AGENT_BRIEF.md" in required
    assert "scripts/agent_intake.py" in required
    assert "docs/04_evaluation_protocols.md" not in required
    assert "agent_protocols/experiment_audit.md" not in required


def test_skill_router_has_context_pack_mapping_for_each_routing_row():
    text = read("agent_protocols/skill_router.md")
    table = text.split("## Routing Table", 1)[1].split("## Required Intake Documents", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "Human scenario" not in line and "---" not in line]
    assert rows
    for row in rows:
        assert "agent_context/" in row or "task-specific pack selected by `scripts/agent_intake.py`" in row
        assert any(mode in row for mode in ["`plan-driven`", "`review-fix`", "`direct-task`", "`planning-only`", "`command-only`", "`state-changing`"])


def test_context_mapping_is_static_for_all_task_families():
    assert TASK_FAMILIES == set(CONTEXT_PACKS)
    for task in TASK_FAMILIES:
        data = context_for_task(task)
        assert data["context_pack_path"].startswith("agent_context/")


def test_explicit_task_family_overrides_keyword_scan():
    result = classify("Task family: context-engineering\nreview PR18 comments")
    assert result["mode"] == "planning-only"
    assert result["task_family"] == "context-engineering"

    result = classify("task_family: context-engineering score-fusion")
    assert result["task_family"] == "context-engineering"


def test_json_output_is_pure_json_for_intake_and_context():
    intake = run_script(
        "scripts/agent_intake.py",
        "--deterministic",
        "--json",
        "--message",
        "score-fusion：修 component map 的 whitening 绑定问题",
    )
    assert json.loads(intake.stdout)["task_family"] == "score-fusion"
    context = run_script("scripts/agent_context.py", "--json", "--task", "score-fusion")
    assert json.loads(context.stdout)["context_pack_path"] == "agent_context/score_fusion_pack.md"


def test_review_inbox_creates_one_issue_per_unresolved_thread(monkeypatch, tmp_path):
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [
                            {
                                "id": "THREAD_alpha",
                                "isResolved": False,
                                "isOutdated": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "body": "Please keep --json output parseable.\npython scripts/agent_intake.py --json --message x",
                                            "path": "scripts/agent_intake.py",
                                            "line": 450,
                                            "author": {"login": "reviewer"},
                                            "createdAt": "2026-05-17T00:00:00Z",
                                        }
                                    ]
                                },
                            },
                            {
                                "id": "THREAD_beta",
                                "isResolved": False,
                                "isOutdated": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "body": "Read-only PR review must not be editable.",
                                            "path": "scripts/agent_intake.py",
                                            "line": 120,
                                            "author": {"login": "reviewer"},
                                            "createdAt": "2026-05-17T00:01:00Z",
                                        }
                                    ]
                                },
                            },
                        ]
                    },
                    "comments": {"nodes": []},
                }
            }
        }
    }

    monkeypatch.setattr("scripts.agent_review_inbox.fetch_pr_review_threads", lambda pr: payload)
    issues = ingest_review_inbox(18, tmp_path)
    board = json.loads((tmp_path / "issue_board.json").read_text(encoding="utf-8"))
    assert len(issues) == 2
    assert len(board["issues"]) == 2
    assert {issue["thread_id"] for issue in board["issues"]} == {"THREAD_alpha", "THREAD_beta"}
    assert {issue["line_hint"] for issue in board["issues"]} == {120, 450}
    assert all(issue["files_hint"] == ["scripts/agent_intake.py"] for issue in board["issues"])


def test_inline_review_thread_preserves_metadata_and_strips_useful_suffix(monkeypatch, tmp_path):
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [
                            {
                                "id": "THREAD_p1",
                                "isResolved": False,
                                "isOutdated": True,
                                "comments": {
                                    "nodes": [
                                        {
                                            "body": "[P1] JSON output should stay parseable.\n\nUseful? React with 👍 / 👎",
                                            "path": "scripts/agent_intake.py",
                                            "line": 123,
                                            "author": {"login": "chatgpt-codex-connector"},
                                            "createdAt": "2026-05-17T00:00:00Z",
                                        }
                                    ]
                                },
                            },
                            {
                                "id": "THREAD_resolved",
                                "isResolved": True,
                                "isOutdated": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "body": "[P2] Already resolved.",
                                            "path": "scripts/agent_intake.py",
                                            "line": 456,
                                            "author": {"login": "chatgpt-codex-connector"},
                                            "createdAt": "2026-05-17T00:01:00Z",
                                        }
                                    ]
                                },
                            },
                        ]
                    },
                    "comments": {"nodes": []},
                }
            }
        }
    }

    monkeypatch.setattr("scripts.agent_review_inbox.fetch_pr_review_threads", lambda pr: payload)
    issues = ingest_review_inbox(18, tmp_path)
    assert len(issues) == 1
    issue = issues[0]
    assert issue["kind"] == "inline_thread"
    assert issue["thread_id"] == "THREAD_p1"
    assert issue["line_hint"] == 123
    assert issue["files_hint"] == ["scripts/agent_intake.py"]
    assert "Useful? React" not in issue["problem"]
    assert issue["stale"] is True
    assert issue["outdated"] is True


def test_real_inline_badge_markdown_maps_to_expected_severity(monkeypatch, tmp_path):
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [
                            {
                                "id": "THREAD_badge_p1",
                                "isResolved": False,
                                "isOutdated": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "body": "**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub> JSON output issue**",
                                            "path": "scripts/agent_intake.py",
                                            "line": 99,
                                            "author": {"login": "chatgpt-codex-connector"},
                                            "createdAt": "2026-05-17T00:00:00Z",
                                        }
                                    ]
                                },
                            },
                            {
                                "id": "THREAD_badge_p2",
                                "isResolved": False,
                                "isOutdated": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "body": "**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub> Medium issue**",
                                            "path": "scripts/agent_review_inbox.py",
                                            "line": 101,
                                            "author": {"login": "chatgpt-codex-connector"},
                                            "createdAt": "2026-05-17T00:01:00Z",
                                        }
                                    ]
                                },
                            },
                            {
                                "id": "THREAD_badge_p3",
                                "isResolved": False,
                                "isOutdated": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "body": "**<sub><sub>![P3 Badge](https://img.shields.io/badge/P3-blue?style=flat)</sub></sub> Low issue**",
                                            "path": "scripts/agent_review_inbox.py",
                                            "line": 102,
                                            "author": {"login": "chatgpt-codex-connector"},
                                            "createdAt": "2026-05-17T00:02:00Z",
                                        }
                                    ]
                                },
                            },
                        ]
                    },
                    "comments": {"nodes": []},
                }
            }
        }
    }

    monkeypatch.setattr("scripts.agent_review_inbox.fetch_pr_review_threads", lambda pr: payload)
    issues = ingest_review_inbox(18, tmp_path)
    by_thread = {issue["thread_id"]: issue["severity"] for issue in issues}
    assert by_thread["THREAD_badge_p1"] == "S0"
    assert by_thread["THREAD_badge_p2"] == "S1"
    assert by_thread["THREAD_badge_p3"] == "S2"


def test_copilot_digest_comment_splits_multiple_issues():
    body = """> @copilot review

### 🔴 高风险

**1. `scripts/agent_intake.py`：JSON 输出被破坏**

需要保持 `--json` 为纯 JSON。
python -m pytest tests/test_agent_intake_workflow.py -q

### 🟠 中风险

> **⚠️ 中级风险：session ledger 没记录命令**

涉及 `scripts/agent_session.py`。
"""
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {"nodes": []},
                    "comments": {
                        "nodes": [
                            {
                                "id": "COPILOT_digest",
                                "url": "https://example.test/comment/1",
                                "body": body,
                                "author": {"login": "Copilot"},
                                "createdAt": "2026-05-17T00:00:00Z",
                            }
                        ]
                    },
                }
            }
        }
    }

    issues = issues_from_payload(payload, 18)
    assert len(issues) >= 2
    assert {issue["kind"] for issue in issues} == {"copilot_digest_comment"}
    assert issues[0]["severity"] == "S0"
    assert any("scripts/agent_intake.py" in issue["files_hint"] for issue in issues)
    assert any("scripts/agent_session.py" in issue["files_hint"] for issue in issues)
    assert any(issue["validation_hint"] for issue in issues)


def test_copilot_digest_without_new_blockers_creates_no_issue():
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {"nodes": []},
                    "comments": {
                        "nodes": [
                            {
                                "id": "COPILOT_clean",
                                "body": "> @copilot review\n\n当前未发现新的阻塞问题，可合入。",
                                "author": {"login": "Copilot"},
                                "createdAt": "2026-05-17T00:00:00Z",
                            }
                        ]
                    },
                }
            }
        }
    }

    assert issues_from_payload(payload, 18) == []


def test_trigger_and_human_fix_report_comments_create_no_issue():
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {"nodes": []},
                    "comments": {
                        "nodes": [
                            {"id": "trigger", "body": "@copilot review", "author": {"login": "metaphordivine"}},
                            {"id": "fixed", "body": "已处理这轮 review，验证通过。", "author": {"login": "metaphordivine"}},
                        ]
                    },
                }
            }
        }
    }

    assert classify_pr_comment(payload["data"]["repository"]["pullRequest"]["comments"]["nodes"][0]) == "trigger_comment"
    assert classify_pr_comment(payload["data"]["repository"]["pullRequest"]["comments"]["nodes"][1]) == "human_fix_report"
    assert issues_from_payload(payload, 18, include_pr_comments=True) == []


def test_copilot_digest_deepseek_engine_can_split(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            body = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "issues": [
                                        {
                                            "severity": "S1",
                                            "title": "DeepSeek split issue",
                                            "problem": "`scripts/agent_review_inbox.py` should split digest comments.",
                                            "files_hint": ["scripts/agent_review_inbox.py"],
                                            "validation_hint": ["python -m pytest tests/test_agent_intake_workflow.py -q"],
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
            return json.dumps(body).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("scripts.agent_review_inbox.urllib.request.urlopen", fake_urlopen)
    issues = parse_copilot_digest(
        "### 🟠 中风险\n`x.py` digest issue",
        18,
        "COMMENT_deepseek",
        digest_engine="deepseek",
        deepseek_api_key="test-key",
        require_deepseek=True,
    )

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["payload"]["model"] == "deepseek-v4-flash"
    assert issues[0]["source"].endswith(":deepseek")
    assert issues[0]["files_hint"] == ["scripts/agent_review_inbox.py"]


def test_review_digest_uses_deepseek_from_user_env_when_process_env_missing(monkeypatch):
    captured = {}

    def fake_windows_env_value(name, scope):
        if scope == "User" and name == "AGENT_REVIEW_DIGEST_ENGINE":
            return "deepseek"
        return None

    def fake_parse(body, pr, comment_ref, *, api_key, model, base_url, timeout):
        captured["api_key"] = api_key
        return [
            {
                "id": "PR18-COPILOT-deepseek-1",
                "source": f"pr:{pr}:copilot_digest:{comment_ref}:deepseek",
                "kind": "copilot_digest_comment",
                "severity": "S1",
                "title": "DeepSeek user-env issue",
                "problem": body,
                "expected_fix": "",
                "files_hint": [],
                "validation_hint": [],
                "status": "open",
            }
        ]

    monkeypatch.delenv("AGENT_REVIEW_DIGEST_ENGINE", raising=False)
    monkeypatch.setenv("AGENT_ENABLE_EXTERNAL_ENV_LOOKUP_FOR_TESTS", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("scripts.agent_env._windows_env_value", fake_windows_env_value)
    monkeypatch.setattr("scripts.agent_review_inbox.parse_copilot_digest_comment_with_deepseek", fake_parse)

    issues = parse_copilot_digest("### 🟠 中风险\n`x.py` digest issue", 18, "COMMENT_deepseek")

    assert captured["api_key"] == "test-key"
    assert issues[0]["source"].endswith(":deepseek")


def test_review_file_copilot_digest_uses_digest_parser(monkeypatch, tmp_path):
    review_file = tmp_path / "copilot.md"
    review_file.write_text(
        """> @copilot review

### 🔴 高风险

**1. `scripts/agent_intake.py`：JSON 输出被破坏**

需要保持 `--json` 为纯 JSON。

### 🟠 中风险

**2. `scripts/agent_review_inbox.py`：digest fallback 合并问题**

需要拆成多个 issue。
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.agent_review_inbox.fetch_pr_review_threads",
        lambda pr: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    issues = ingest_review_inbox(18, tmp_path / "out", review_file=review_file)
    assert looks_like_copilot_digest(review_file.read_text(encoding="utf-8")) is True
    assert len(issues) >= 2
    assert {issue["kind"] for issue in issues} == {"copilot_digest_comment"}
    assert any("scripts/agent_intake.py" in issue["files_hint"] for issue in issues)
    assert any("scripts/agent_review_inbox.py" in issue["files_hint"] for issue in issues)


def test_review_file_structured_plan_still_uses_plan_parser(monkeypatch, tmp_path):
    review_file = tmp_path / "plan.md"
    review_file.write_text(
        """
### S0-1 first issue
Problem in scripts/agent_intake.py.

### S0-2 second issue
Problem in scripts/agent_review_inbox.py.
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.agent_review_inbox.fetch_pr_review_threads",
        lambda pr: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    issues = ingest_review_inbox(18, tmp_path / "out", review_file=review_file)
    assert looks_like_copilot_digest(review_file.read_text(encoding="utf-8")) is False
    assert {issue["id"] for issue in issues} == {"S0-1", "S0-2"}


def test_review_inbox_filters_pr_level_comments_by_default():
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {"nodes": []},
                    "comments": {
                        "nodes": [
                            {"body": "Looks good to me.", "createdAt": "2026-05-17T00:00:00Z"},
                            {"body": "Please fix the JSON output.", "createdAt": "2026-05-17T00:01:00Z"},
                        ]
                    },
                }
            }
        }
    }

    default_issues = issues_from_payload(payload, 18)
    included_issues = issues_from_payload(payload, 18, include_pr_comments=True)
    assert [issue["title"] for issue in default_issues] == ["Please fix the JSON output."]
    assert len(included_issues) == 2


def test_p2_p3_severity_mapping_is_consistent():
    plan_issues = parse_text("### P2 medium issue\nbody\n\n### P3 low issue\nbody", source="test")
    assert [issue["severity"] for issue in plan_issues] == ["S1", "S2"]
    assert infer_review_severity("[P2] please fix this") == "S1"
    assert infer_review_severity("[P3] optional follow-up") == "S2"


def test_agent_brief_warns_about_deepseek_private_review_content():
    text = read("agent_protocols/AGENT_BRIEF.md")
    assert "may transmit user task text to the external DeepSeek API" in text
    assert "private review content" in text


def test_deepseek_refinement_uses_local_schema_and_flags(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            body = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "mode": "direct-task",
                                    "task_family": "score-fusion",
                                    "confidence": 0.97,
                                    "reason": "DeepSeek refined a narrow score-fusion request.",
                                }
                            )
                        }
                    }
                ]
            }
            return json.dumps(body).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setattr("scripts.agent_intake.urllib.request.urlopen", fake_urlopen)
    deterministic = classify("score-fusion：修 component map 的 whitening 绑定问题")
    result = classify_with_deepseek(
        "score-fusion：修 component map 的 whitening 绑定问题",
        deterministic,
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        timeout=3,
    )

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["timeout"] == 3
    assert captured["payload"]["model"] == "deepseek-v4-flash"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert captured["authorization"] == "Bearer test-key"
    assert result["mode"] == "direct-task"
    assert result["task_family"] == "score-fusion"
    assert result["context_pack"] == "agent_context/score_fusion_pack.md"
    assert result["should_edit"] is True
    assert result["deepseek"]["used"] is True


def test_default_intake_does_not_call_deepseek_when_key_exists(monkeypatch, capsys):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("DeepSeek should not be called by default")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_INTAKE_ENGINE", raising=False)
    monkeypatch.setattr("scripts.agent_intake.classify_with_deepseek", fail_if_called)

    assert intake_main(["--json", "--message", "score-fusion：修 component map 的 whitening 绑定问题"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["task_family"] == "score-fusion"
    assert "deepseek" not in result


def test_intake_uses_deepseek_from_user_env_when_process_env_missing(monkeypatch, capsys):
    def fake_windows_env_value(name, scope):
        if scope == "User" and name == "AGENT_INTAKE_ENGINE":
            return "deepseek"
        return None

    def fake_refine(text, deterministic_result, *, api_key, model, base_url, timeout):
        assert api_key == "test-key"
        refined = dict(deterministic_result)
        refined["deepseek"] = {"used": True, "model": model, "base_url": base_url}
        return refined

    monkeypatch.delenv("AGENT_INTAKE_ENGINE", raising=False)
    monkeypatch.setenv("AGENT_ENABLE_EXTERNAL_ENV_LOOKUP_FOR_TESTS", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("scripts.agent_env._windows_env_value", fake_windows_env_value)
    monkeypatch.setattr("scripts.agent_intake.classify_with_deepseek", fake_refine)

    assert intake_main(["--json", "--message", "score-fusion：修 component map 的 whitening 绑定问题"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["deepseek"]["used"] is True


def test_explicit_deepseek_flag_calls_deepseek(monkeypatch, capsys):
    def fake_refine(text, deterministic_result, *, api_key, model, base_url, timeout):
        assert api_key == "test-key"
        refined = dict(deterministic_result)
        refined["confidence"] = 0.96
        refined["deepseek"] = {"used": True, "model": model, "base_url": base_url}
        return refined

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("scripts.agent_intake.classify_with_deepseek", fake_refine)
    assert intake_main(["--deepseek", "--json", "--message", "score-fusion：修 component map 的 whitening 绑定问题"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["deepseek"]["used"] is True
    assert result["confidence"] == 0.96


def test_auto_intake_is_deterministic_without_key():
    deterministic = classify("score-fusion：修 component map 的 whitening 绑定问题")
    result = maybe_refine_with_deepseek(
        "score-fusion：修 component map 的 whitening 绑定问题",
        deterministic,
        engine="auto",
        api_key=None,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        timeout=3,
        require_deepseek=False,
    )

    assert result == deterministic


def test_deepseek_engine_falls_back_to_deterministic_on_request_failure(monkeypatch):
    def fake_refine(text, deterministic_result, *, api_key, model, base_url, timeout):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("scripts.agent_intake.classify_with_deepseek", fake_refine)
    deterministic = classify("score-fusion：修 component map 的 whitening 绑定问题")
    result = maybe_refine_with_deepseek(
        "score-fusion：修 component map 的 whitening 绑定问题",
        deterministic,
        engine="deepseek",
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        timeout=3,
        require_deepseek=False,
    )

    assert result["mode"] == deterministic["mode"]
    assert result["deepseek"]["used"] is False
    assert "used deterministic intake" in result["deepseek"]["fallback"]
