#!/usr/bin/env python3
"""Copy a validated final image to a collision-safe delivery folder."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def available_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    source_name = Path(filename)
    stem = source_name.stem
    suffix = source_name.suffix
    index = 2
    while True:
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a final image to Desktop/images without overwriting files."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=Path.home() / "Desktop" / "images",
    )
    parser.add_argument(
        "--name",
        help="Optional delivery filename, including its extension.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Final image not found: {source}")

    destination_dir = args.destination_dir.expanduser().resolve()
    delivery_name = args.name or source.name
    if Path(delivery_name).name != delivery_name:
        raise SystemExit("--name must be a filename, not a path.")
    if not Path(delivery_name).suffix:
        raise SystemExit("--name must include a file extension.")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = available_path(destination_dir, delivery_name)
    shutil.copy2(source, destination)
    print(f"Delivered: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
