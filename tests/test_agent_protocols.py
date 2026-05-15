from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_foundation_usage_protocol_is_routed_from_required_docs():
    required = "agent_protocols/foundation_usage.md"

    for path in [
        "README.md",
        "AGENTS.md",
        "docs/目录怎么用.md",
        "agent_protocols/skill_router.md",
    ]:
        assert required in read(path)


def test_required_first_reads_keep_foundation_protocol_in_order():
    expected = [
        "README.md",
        "docs/目录怎么用.md",
        "docs/04_evaluation_protocols.md",
        "agent_protocols/experiment_audit.md",
        "agent_protocols/foundation_usage.md",
        "agent_protocols/skill_router.md",
        "AGENTS.md",
    ]
    text = read("agent_protocols/skill_router.md").split("## Required Documents", 1)[1]
    positions = [text.index(item) for item in expected]

    assert positions == sorted(positions)


def test_foundation_usage_protocol_names_stable_foundations_and_commands():
    text = read("agent_protocols/foundation_usage.md")

    required_terms = [
        "Dataset Manifest Builder",
        "Split Manifest Builder",
        "Deterministic Utilities",
        "Run Manifest Writer",
        "Protocol Runner Skeleton",
        "Component Artifact Contract",
        "Prediction Writer",
        "Metric Report Builder",
        "No-Leakage Source Scanner",
        "Promotion Audit Helper",
        "Registry Consistency",
        "Training Monitor",
        "Cache Manager",
    ]
    for term in required_terms:
        assert term in text

    required_commands = [
        "scripts/build_dataset_manifest.py",
        "scripts/build_split_manifest.py",
        "scripts/data_qa_report.py",
        "scripts/plan_evaluation_protocol.py",
        "scripts/run_evaluation_protocol.py",
        "scripts/assemble_score_route.py",
        "scripts/scan_no_leakage.py",
        "scripts/check_promotion_audit.py",
        "scripts/check_registry_consistency.py",
        "scripts/generate_component_docs.py",
        "python scripts/repo_doctor.py fast",
    ]
    for command in required_commands:
        assert command in text

    required_apis = [
        "hust_bci_er.data.manifest_builder",
        "hust_bci_er.data.split_builder",
        "hust_bci_er.training.reproducibility",
        "hust_bci_er.audit.run_manifest.write_run_manifest",
        "hust_bci_er.contracts.artifacts",
        "hust_bci_er.evaluation.prediction_writer",
        "hust_bci_er.inference.score_route_assembly",
        "hust_bci_er.evaluation.report",
        "hust_bci_er.audit.source_scanner",
        "hust_bci_er.training.monitor",
        "hust_bci_er.audit.cache_manager",
    ]
    for api in required_apis:
        assert api in text


def test_foundation_usage_protocol_preserves_audit_red_lines():
    text = read("agent_protocols/foundation_usage.md")

    required_red_lines = [
        "The runner manifest is a job plan and lock. It is not a prediction table, score",
        "Do not claim candidate readiness if the dataset/split manifest is still a",
        "Do not treat `protocol_run_manifest.json` as training evidence.",
        "Do not commit checkpoints, raw outputs, full prediction tables, caches, or temporary dumps.",
        "not model features",
        "keep the route status unchanged",
    ]
    for red_line in required_red_lines:
        assert red_line in text


def test_foundation_artifact_router_does_not_overlap_score_route_trigger():
    text = read("agent_protocols/skill_router.md")

    score_route_line = next(line for line in text.splitlines() if "Score Route Assembly Skill" in line)
    foundation_artifact_line = next(line for line in text.splitlines() if "Foundation Artifact Skill" in line)

    assert "assemble component scores" in score_route_line
    assert "assemble component scores" not in foundation_artifact_line


def test_foundation_usage_protocol_references_existing_script_entrypoints():
    scripts = [
        "scripts/build_dataset_manifest.py",
        "scripts/build_split_manifest.py",
        "scripts/data_qa_report.py",
        "scripts/plan_evaluation_protocol.py",
        "scripts/run_evaluation_protocol.py",
        "scripts/assemble_score_route.py",
        "scripts/scan_no_leakage.py",
        "scripts/check_promotion_audit.py",
        "scripts/check_registry_consistency.py",
        "scripts/generate_component_docs.py",
        "scripts/generate_route_summary.py",
        "scripts/repo_doctor.py",
    ]
    for script in scripts:
        assert (ROOT / script).exists(), script


def test_skill_router_exposes_foundation_skills():
    text = read("agent_protocols/skill_router.md")

    for skill in [
        "Foundation Usage Skill",
        "Foundation Evidence Skill",
        "Foundation Artifact Skill",
        "Foundation Maintenance Skill",
    ]:
        assert skill in text

    assert "agent_protocols/foundation_usage.md" in text
