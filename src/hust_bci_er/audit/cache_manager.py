"""Local cache manifest manager for derived artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from hust_bci_er.audit.manifest import sha256_file


class CacheManager:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "cache_manifest.json"

    def load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"schema": "cache_manifest_v1", "entries": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save(self, manifest: Mapping[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def register(self, *, key: str, path: Path, kind: str, source_sha256: str | None = None) -> dict[str, Any]:
        resolved = path.resolve()
        try:
            relative_path = resolved.relative_to(self.cache_dir)
        except ValueError as exc:
            raise ValueError(f"cache artifact must stay under cache_dir: {path}") from exc
        entry = {
            "key": key,
            "path": relative_path.as_posix(),
            "kind": kind,
            "sha256": sha256_file(resolved),
            "source_sha256": source_sha256,
        }
        manifest = self.load()
        entries = [item for item in manifest.get("entries", []) if item.get("key") != key]
        entries.append(entry)
        manifest["entries"] = entries
        self.save(manifest)
        return entry
