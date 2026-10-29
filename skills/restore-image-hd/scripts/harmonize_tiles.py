#!/usr/bin/env python3
"""Match redraw tiles' low-frequency luminance and chroma to source tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageFilter
except ImportError as exc:
    raise SystemExit(
        "Pillow and NumPy are required. Use the Codex bundled Python runtime."
    ) from exc


def blurred_ycbcr(image: Image.Image, radius: float) -> np.ndarray:
    channels = [
        np.asarray(channel.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        for channel in image.split()
    ]
    return np.stack(channels, axis=-1)


def harmonize(
    source_path: Path,
    redraw_path: Path,
    output_path: Path,
    blur_radius: float,
    luma_limit: float,
    chroma_limit: float,
) -> dict[str, object]:
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    with Image.open(redraw_path) as opened:
        alpha = opened.getchannel("A") if opened.mode == "RGBA" else None
        redraw = opened.convert("RGB")
    if source.size != redraw.size:
        raise SystemExit(f"Size mismatch: {source_path.name} and {redraw_path.name}")

    source_ycc = source.convert("YCbCr")
    redraw_ycc = redraw.convert("YCbCr")
    correction = blurred_ycbcr(source_ycc, blur_radius) - blurred_ycbcr(
        redraw_ycc,
        blur_radius,
    )
    correction[..., 0] = np.clip(correction[..., 0], -luma_limit, luma_limit)
    correction[..., 1:] = np.clip(
        correction[..., 1:],
        -chroma_limit,
        chroma_limit,
    )
    redraw_pixels = np.asarray(redraw_ycc, dtype=np.float32)
    corrected = np.clip(redraw_pixels + correction, 0, 255).astype(np.uint8)
    result = Image.fromarray(corrected, "YCbCr").convert("RGB")
    if alpha is not None:
        result.putalpha(alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)

    mean_delta = correction.reshape(-1, 3).mean(axis=0)
    return {
        "file": redraw_path.name,
        "mean_applied_ycbcr": [round(float(value), 3) for value in mean_delta],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use matching source tiles as low-frequency color anchors before "
            "merging independently redrawn tiles."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("redraw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--blur-radius", type=float, default=96)
    parser.add_argument("--luma-limit", type=float, default=28)
    parser.add_argument("--chroma-limit", type=float, default=18)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.blur_radius <= 0:
        raise SystemExit("--blur-radius must be positive.")
    if args.luma_limit < 0 or args.chroma_limit < 0:
        raise SystemExit("Color correction limits must be non-negative.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = []
    for tile in manifest["tiles"]:
        filename = tile["file"]
        report.append(
            harmonize(
                args.source_dir / filename,
                args.redraw_dir / filename,
                args.output_dir / filename,
                args.blur_radius,
                args.luma_limit,
                args.chroma_limit,
            )
        )
    report_path = args.output_dir / "harmonization.json"
    report_path.write_text(
        json.dumps(
            {
                "blur_radius": args.blur_radius,
                "luma_limit": args.luma_limit,
                "chroma_limit": args.chroma_limit,
                "tiles": report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Harmonized {len(report)} tiles: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
