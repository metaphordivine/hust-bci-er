from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"
DEFAULT_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
DEFAULT_OUTPUT = ROOT / "scratch" / "model_weights" / "cbramod" / "pretrained_weights.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    try:
        with urlopen(url, timeout=60) as response, tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(output)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify the author-released CBraMod pretrained weights.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if not output.exists() or args.force:
        download(str(args.url), output)
    actual = sha256_file(output)
    if actual != str(args.sha256).lower():
        print(f"sha256 mismatch for {output}: expected {args.sha256}, got {actual}", file=sys.stderr)
        return 1
    print(f"OK {output} sha256={actual} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
