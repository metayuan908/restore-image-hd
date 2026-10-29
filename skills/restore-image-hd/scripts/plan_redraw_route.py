#!/usr/bin/env python3
"""Choose a conservative redraw route before any GPT Image requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required. Use the Codex bundled Python runtime.") from exc


PRESETS = {
    "4k-landscape": (3840, 2160),
    "4k-portrait": (2160, 3840),
    "4k-square": (4096, 4096),
    "8k-landscape": (7680, 4320),
    "8k-portrait": (4320, 7680),
    "8k-square": (7680, 7680),
}

MODE_SCORE = {
    "texture": 0,
    "background": 0,
    "scenery": 0,
    "abstract": 0,
    "general": 1,
    "product": 3,
    "people": 3,
    "architecture": 3,
    "poster": 5,
    "packaging": 5,
    "text": 5,
    "ui": 5,
    "diagram": 5,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score semantic and structural redraw risk, then recommend direct tiles, "
            "a global coherence anchor, or mask-driven segmented redraw."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--content-mode",
        choices=tuple(MODE_SCORE),
        default="general",
    )
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--connected-subjects", action="store_true")
    parser.add_argument("--precise-geometry", action="store_true")
    parser.add_argument("--has-text", action="store_true")
    parser.add_argument("--sensitive-text", action="store_true")
    parser.add_argument(
        "--background-only",
        action="store_true",
        help="Remove text and redraw imagery only; do not generate typography.",
    )
    parser.add_argument("--write-json", type=Path)
    return parser


def resolve_target(
    args: argparse.Namespace, source_width: int, source_height: int
) -> tuple[int, int]:
    if (args.target_width is None) != (args.target_height is None):
        raise SystemExit("Pass both --target-width and --target-height.")
    if args.target_width is not None:
        target = (args.target_width, args.target_height)
    elif args.preset:
        target = PRESETS[args.preset]
    else:
        target = (source_width, source_height)
    if target[0] <= 0 or target[1] <= 0:
        raise SystemExit("Target dimensions must be positive.")
    return target


def main() -> int:
    args = build_parser().parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input image not found: {source}")
    if args.sensitive_text:
        args.has_text = True

    with Image.open(source) as opened:
        source_width, source_height = opened.size
        mode = opened.mode
        has_alpha = "A" in opened.getbands() and opened.getchannel("A").getextrema() != (
            255,
            255,
        )
    target_width, target_height = resolve_target(args, source_width, source_height)
    scale = max(target_width / source_width, target_height / source_height)

    score = MODE_SCORE[args.content_mode]
    reasons = [f"content mode {args.content_mode}: +{score}"]
    if args.connected_subjects:
        score += 3
        reasons.append("connected subject crosses possible crop boundaries: +3")
    if args.precise_geometry:
        score += 2
        reasons.append("precise geometry or repeated alignment: +2")
    if args.sensitive_text:
        score += 3
        reasons.append("exact text, logo, price, UI, or code: +3")
    elif args.has_text:
        score += 1
        reasons.append("visible text requires separation: +1")
    if args.background_only:
        score += 2
        reasons.append("background-only output requires text-removal masks: +2")
    if scale >= 4:
        score += 1
        reasons.append(f"large delivery scale {scale:.2f}x: +1")
    if has_alpha:
        score += 1
        reasons.append("source alpha must be preserved deterministically: +1")

    if score >= 5:
        risk = "high"
        route = "segmented"
        command = "gpt_segmented_redraw.py"
        pilot_count = 2
    elif score >= 2:
        risk = "medium"
        route = "coherent"
        command = "gpt_coherent_redraw.py"
        pilot_count = 2 if args.connected_subjects else 1
    else:
        risk = "low"
        route = "direct-tiles"
        command = "gpt_tile_redraw.py"
        pilot_count = 0

    if args.background_only:
        text_action = "mask-and-remove; do not ask GPT to render typography"
    elif args.sensitive_text or args.has_text:
        text_action = "protect or rebuild deterministically at delivery size"
    else:
        text_action = "none detected by caller"

    report = {
        "input": str(source),
        "source_dimensions": [source_width, source_height],
        "source_mode": mode,
        "has_alpha": has_alpha,
        "target_dimensions": [target_width, target_height],
        "delivery_scale": scale,
        "content_mode": args.content_mode,
        "risk_score": score,
        "risk": risk,
        "route": route,
        "recommended_script": command,
        "pilot_regions": pilot_count,
        "text_action": text_action,
        "reasons": reasons,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.write_json:
        destination = args.write_json.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
