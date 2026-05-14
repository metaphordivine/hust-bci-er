"""Generate lightweight component documentation from registries."""

from __future__ import annotations

from hust_bci_er.config import registry


def render_component_docs() -> str:
    sections = [
        ("Preprocessing", registry.PREPROCESSING),
        ("Features", registry.FEATURES),
        ("Torch Backbones", registry.TORCH_BACKBONES),
        ("Sklearn Models", registry.SKLEARN_MODELS),
        ("Graph Models", registry.GRAPH_MODELS),
        ("Score Route Models", registry.SCORE_ROUTE_MODELS),
        ("Adaptation", registry.ADAPTATION),
        ("Evaluation Protocols", registry.EVALUATION_PROTOCOLS),
        ("Primary Metrics", registry.PRIMARY_METRICS),
    ]
    lines = ["# Component Registry", ""]
    for title, values in sections:
        lines.extend([f"## {title}", ""])
        for value in sorted(values):
            lines.append(f"- `{value}`")
        lines.append("")
    return "\n".join(lines)
