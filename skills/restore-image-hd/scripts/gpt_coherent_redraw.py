#!/usr/bin/env python3
"""Create a global GPT structure anchor, then redraw and stitch native-detail tiles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
MIN_MODEL_PIXELS = 655_360
MAX_MODEL_PIXELS = 8_294_400
MAX_MODEL_EDGE = 3840


def run_command(
    command: list[str], environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, env=environment)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(message)
    return result


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if not environment.get("SSL_CERT_FILE"):
        try:
            import certifi

            environment["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass
    return environment


def find_gpt_cli(explicit: str | None) -> Path:
    sibling = (
        Path(__file__).resolve().parents[2]
        / "codex-gpt-image"
        / "scripts"
        / "codex_gpt_image.py"
    )
    candidates = [
        explicit,
        os.environ.get("CODEX_GPT_IMAGE_CLI"),
        str(sibling),
        str(Path.home() / ".codex/skills/codex-gpt-image/scripts/codex_gpt_image.py"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser().resolve()
    raise SystemExit("codex-gpt-image CLI not found. Install it or pass --gpt-cli.")


def largest_model_size(target_width: int, target_height: int) -> tuple[int, int]:
    ratio = target_width / target_height
    best: tuple[int, int, int, float] | None = None
    for width in range(16, MAX_MODEL_EDGE + 1, 16):
        height = round((width / ratio) / 16) * 16
        if height < 16 or height > MAX_MODEL_EDGE:
            continue
        pixels = width * height
        if not MIN_MODEL_PIXELS <= pixels <= MAX_MODEL_PIXELS:
            continue
        ratio_error = abs(width / height - ratio) / ratio
        candidate = (pixels, width, height, ratio_error)
        if best is None or candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[3] < best[3]
        ):
            best = candidate
    if best is None:
        raise SystemExit("Cannot choose a valid GPT Image size for this aspect ratio.")
    return best[1], best[2]


def read_instruction(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.prompt_file:
        prompt_path = args.prompt_file.expanduser().resolve()
        if not prompt_path.is_file():
            raise SystemExit(f"Prompt file not found: {prompt_path}")
        parts.append(prompt_path.read_text(encoding="utf-8").strip())
    if args.instruction:
        parts.append(args.instruction.strip())
    return "\n".join(part for part in parts if part)


def mode_guidance(mode: str) -> str:
    return {
        "product": (
            "Keep one physically coherent product. Lock perspective, silhouette, "
            "component layout, frame thickness, bevel radius, assembly gaps, "
            "occlusion, material boundaries, roughness, reflections, refraction, "
            "and edge-highlight continuity. Do not melt or duplicate components."
        ),
        "people": (
            "Preserve every identity, person count, pose, clothing, gaze, and "
            "relationship. Keep each hand attached to the correct owner; preserve "
            "wrist, palm, thumb, finger count, grip, contact, and limb continuity."
        ),
        "architecture": (
            "Lock global perspective, vanishing points, straight lines, repeating "
            "modules, structural joints, scale, materials, and shadow direction."
        ),
        "general": (
            "Lock the complete composition, object count, geometry, perspective, "
            "spatial relationships, materials, lighting, and shadows."
        ),
    }[mode]


def build_prompt(mode: str, instruction: str, reference_count: int) -> str:
    reference_note = (
        f"Images 2 through {reference_count + 1} are supporting references only. "
        "Use only explicitly requested material or detail qualities; never copy "
        "their objects, crop, branding, text, or layout."
        if reference_count
        else ""
    )
    return f"""Create a global structure anchor for later overlapping-tile redraw.
Image 1 is authoritative. Reconstruct the complete image in one coherent pass.
{mode_guidance(mode)}
Preserve crop and aspect ratio. Do not add, remove, duplicate, reframe, zoom,
translate, or invent subjects. Do not add a watermark or new text.
{reference_note}
User-specific guidance:
{instruction or "Improve faithful clarity and natural material definition only."}
"""


def crop_anchor_to_ratio(
    source: Path,
    target_ratio: float,
    output: Path,
    max_drift: float,
) -> tuple[int, int, bool]:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    native_size = image.size
    source_ratio = image.width / image.height
    drift = abs(source_ratio - target_ratio) / target_ratio
    if drift > max_drift:
        raise SystemExit(
            f"Global anchor aspect ratio drifted by {drift:.1%}, above "
            f"--max-anchor-aspect-drift {max_drift:.1%}. Refusing to stretch it."
        )
    cropped = False
    if source_ratio > target_ratio:
        width = round(image.height * target_ratio)
        left = (image.width - width) // 2
        image = image.crop((left, 0, left + width, image.height))
        cropped = True
    elif source_ratio < target_ratio:
        height = round(image.width / target_ratio)
        top = (image.height - height) // 2
        image = image.crop((0, top, image.width, top + height))
        cropped = True
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return native_size[0], native_size[1], cropped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use GPT Image for a global structure anchor, then redraw overlapping "
            "native-detail tiles and stitch the requested 4K/8K canvas. No "
            "Real-ESRGAN or detail-upscaling engine is used."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--content-mode",
        choices=("product", "people", "architecture", "general"),
        required=True,
    )
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--instruction")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--reference", action="append", type=Path, default=[])
    parser.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="high")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--gpt-cli")
    parser.add_argument("--tile-width", type=int)
    parser.add_argument("--tile-height", type=int)
    parser.add_argument("--overlap-x", type=int)
    parser.add_argument("--overlap-y", type=int)
    parser.add_argument("--max-tiles", type=int, default=64)
    parser.add_argument("--max-anchor-aspect-drift", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument("--allow-aspect-change", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input image not found: {source}")
    if output.exists() and not args.force_output:
        raise SystemExit(f"Output already exists: {output}. Pass --force-output.")
    if (args.target_width is None) != (args.target_height is None):
        raise SystemExit("Pass both --target-width and --target-height.")
    if args.preset and args.target_width is not None:
        raise SystemExit("Choose a preset or custom dimensions, not both.")
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative.")
    if not 0 <= args.max_anchor_aspect_drift <= 0.25:
        raise SystemExit("--max-anchor-aspect-drift must be between 0 and 0.25.")

    with Image.open(source) as opened:
        source_dimensions = opened.size
    target_dimensions = (
        (args.target_width, args.target_height)
        if args.target_width is not None
        else PRESETS[args.preset or "4k-landscape"]
    )
    if target_dimensions[0] <= 0 or target_dimensions[1] <= 0:
        raise SystemExit("Target dimensions must be positive.")
    source_ratio = source_dimensions[0] / source_dimensions[1]
    target_ratio = target_dimensions[0] / target_dimensions[1]
    ratio_difference = abs(source_ratio - target_ratio) / source_ratio
    if ratio_difference > 0.01 and not args.allow_aspect_change:
        raise SystemExit(
            "Source and target aspect ratios differ by more than 1%. Crop or extend "
            "the source first; do not stretch a structural image."
        )

    references = [path.expanduser().resolve() for path in args.reference]
    for reference in references:
        if not reference.is_file():
            raise SystemExit(f"Reference image not found: {reference}")
    request_dimensions = largest_model_size(*target_dimensions)
    instruction = read_instruction(args)
    prompt = build_prompt(args.content_mode, instruction, len(references))
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output.parent / f".{output.stem}-gpt-coherent"
    )
    anchor_raw = work_dir / "global-anchor-raw.png"
    anchor = work_dir / "global-anchor.png"
    tile_work_dir = work_dir / "tiles"

    print(
        f"Plan: {source_dimensions[0]}x{source_dimensions[1]} -> global GPT anchor "
        f"{request_dimensions[0]}x{request_dimensions[1]} request -> "
        f"GPT tile redraw and stitch {target_dimensions[0]}x{target_dimensions[1]}; "
        "no super-resolution engine"
    )

    gpt_cli = find_gpt_cli(args.gpt_cli)
    environment = subprocess_environment()
    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            auth = run_command([sys.executable, str(gpt_cli), "auth-status"], environment)
        except RuntimeError as exc:
            raise SystemExit(f"Codex OAuth check failed: {exc}") from exc
        if auth.stdout.strip():
            print(auth.stdout.strip())
        if not (args.resume and anchor_raw.is_file()):
            command = [
                sys.executable,
                str(gpt_cli),
                "generate",
                "--prompt",
                prompt,
                "--image",
                str(source),
            ]
            for reference in references:
                command.extend(["--image", str(reference)])
            command.extend(
                [
                    "--size",
                    f"{request_dimensions[0]}x{request_dimensions[1]}",
                    "--quality",
                    args.quality,
                    "--output-format",
                    "png",
                    "--out",
                    str(anchor_raw),
                    "--timeout",
                    "300",
                ]
            )
            error: RuntimeError | None = None
            for attempt in range(args.retries + 1):
                try:
                    result = run_command(command, environment)
                    if result.stderr.strip():
                        print(result.stderr.strip())
                    error = None
                    break
                except RuntimeError as exc:
                    error = exc
                    if attempt < args.retries:
                        print(f"Retrying global anchor after error: {exc}", flush=True)
                        time.sleep(1)
            if error:
                raise SystemExit(f"Global anchor redraw failed: {error}")
        native_width, native_height, cropped = crop_anchor_to_ratio(
            anchor_raw,
            target_ratio,
            anchor,
            args.max_anchor_aspect_drift,
        )
    else:
        native_width, native_height, cropped = source_dimensions[0], source_dimensions[1], False

    tile_script = Path(__file__).resolve().with_name("gpt_tile_redraw.py")
    tile_command = [
        sys.executable,
        str(tile_script),
        str(source),
        str(output),
        "--content-mode",
        args.content_mode,
        "--allow-structural-tiling",
        "--context-image",
        str(source if args.dry_run else anchor),
        "--target-width",
        str(target_dimensions[0]),
        "--target-height",
        str(target_dimensions[1]),
        "--quality",
        args.quality,
        "--work-dir",
        str(tile_work_dir),
        "--gpt-cli",
        str(gpt_cli),
        "--max-tiles",
        str(args.max_tiles),
    ]
    if instruction:
        tile_command.extend(["--instruction", instruction])
    for option, value in (
        ("--tile-width", args.tile_width),
        ("--tile-height", args.tile_height),
        ("--overlap-x", args.overlap_x),
        ("--overlap-y", args.overlap_y),
    ):
        if value is not None:
            tile_command.extend([option, str(value)])
    if args.resume:
        tile_command.append("--resume")
    if args.dry_run:
        tile_command.append("--dry-run")
    if args.force_output:
        tile_command.append("--force-output")
    if args.allow_aspect_change:
        tile_command.append("--allow-aspect-change")
    try:
        tile_result = run_command(tile_command, environment)
    except RuntimeError as exc:
        raise SystemExit(f"Native-detail tile stage failed: {exc}") from exc
    if tile_result.stdout.strip():
        print(tile_result.stdout.strip())
    if args.dry_run:
        print("Dry run complete; no GPT Image requests were sent.")
        return 0

    tile_metadata = json.loads((tile_work_dir / "run.json").read_text(encoding="utf-8"))
    metadata = {
        "input": str(source),
        "output": str(output),
        "content_mode": args.content_mode,
        "source_dimensions": list(source_dimensions),
        "target_dimensions": list(target_dimensions),
        "global_request_dimensions": list(request_dimensions),
        "global_native_dimensions": [native_width, native_height],
        "global_anchor_cropped": cropped,
        "tile_count": tile_metadata["tile_count"],
        "tile_dimensions": tile_metadata["tile_dimensions"],
        "tile_request_dimensions": tile_metadata["gpt_request_dimensions"],
        "tile_native_dimensions": tile_metadata["tile_native_dimensions"],
        "pixel_enlargement_allowed": False,
        "detail_upscaler_used": False,
        "pipeline": "global GPT anchor -> native-detail GPT tiles -> core stitch",
        "references": [str(path) for path in references],
        "model": "gpt-image-2",
        "authentication": "Codex OAuth",
    }
    (work_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Completed coherent GPT redraw and stitch: {output}; "
        f"global native={native_width}x{native_height}; "
        f"final={target_dimensions[0]}x{target_dimensions[1]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
