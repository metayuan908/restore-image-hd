#!/usr/bin/env python3
"""Redraw overlapping image tiles with GPT Image and merge their core regions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError as exc:
    raise SystemExit("Pillow is required. Use the Codex bundled Python runtime.") from exc


PRESETS = {
    "4k-landscape": (3840, 2160),
    "4k-square": (4096, 4096),
    "2k-landscape": (2560, 1440),
    "full-hd": (1920, 1080),
    "4k-portrait": (2160, 3840),
    "8k-landscape": (7680, 4320),
    "8k-portrait": (4320, 7680),
    "8k-square": (7680, 7680),
    "2k-portrait": (1440, 2560),
    "square-2k": (2048, 2048),
    "social-square": (1080, 1080),
    "social-portrait": (1080, 1350),
    "story": (1080, 1920),
    "ultrawide-4k": (3840, 1600),
}
MIN_MODEL_PIXELS = 655_360
MAX_MODEL_PIXELS = 8_294_400
MAX_MODEL_EDGE = 3840


def round_up_16(value: float) -> int:
    return max(16, int((value + 15) // 16 * 16))


def largest_model_size(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    best: tuple[int, int, int, float] | None = None
    for request_width in range(16, MAX_MODEL_EDGE + 1, 16):
        request_height = round((request_width / ratio) / 16) * 16
        if request_height < 16 or request_height > MAX_MODEL_EDGE:
            continue
        pixels = request_width * request_height
        if not MIN_MODEL_PIXELS <= pixels <= MAX_MODEL_PIXELS:
            continue
        ratio_error = abs(request_width / request_height - ratio) / ratio
        candidate = (pixels, request_width, request_height, ratio_error)
        if best is None or candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[3] < best[3]
        ):
            best = candidate
    if best is None:
        raise SystemExit("Cannot choose a valid GPT Image request size for this tile.")
    return best[1], best[2]


def default_tile_geometry(
    target_width: int, target_height: int, structural: bool
) -> tuple[int, int, int, int]:
    if target_width * target_height <= 2_500_000:
        return target_width, target_height, 0, 0
    if structural:
        ratio = target_width / target_height
        if 0.8 <= ratio <= 1.25:
            tile_width = min(target_width, 1280)
            tile_height = min(target_height, round_up_16(tile_width / ratio))
        elif ratio > 1:
            tile_width = min(target_width, 1536)
            tile_height = min(target_height, round_up_16(tile_width / ratio))
        else:
            tile_height = min(target_height, 1536)
            tile_width = min(target_width, round_up_16(tile_height * ratio))
        overlap_x = 0 if tile_width == target_width else round_up_16(tile_width / 8)
        overlap_y = 0 if tile_height == target_height else round_up_16(tile_height / 8)
        return tile_width, tile_height, overlap_x, overlap_y
    overlap_x = round_up_16(target_width / 15)
    overlap_y = round_up_16(target_height / 15)
    tile_width = round_up_16((target_width + overlap_x) / 2)
    tile_height = round_up_16((target_height + overlap_y) / 2)
    return tile_width, tile_height, overlap_x, overlap_y


def axis_seams(tiles: list[dict], axis: str) -> list[int]:
    size_key = "width" if axis == "x" else "height"
    positions = sorted({int(tile[axis]) for tile in tiles})
    extents = {
        position: max(
            int(tile[axis]) + int(tile[size_key])
            for tile in tiles
            if int(tile[axis]) == position
        )
        for position in positions
    }
    return [
        (extents[left_position] + right_position) // 2
        for left_position, right_position in zip(positions, positions[1:])
    ]


def create_seam_mask(
    manifest: dict,
    output: Path,
    band_width_x: int,
    band_width_y: int,
) -> None:
    width = int(manifest["canvas_width"])
    height = int(manifest["canvas_height"])
    mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    for seam in axis_seams(manifest["tiles"], "x"):
        half = max(1, band_width_x // 2)
        draw.rectangle((seam - half, 0, seam + half, height), fill=(0, 0, 0, 0))
    for seam in axis_seams(manifest["tiles"], "y"):
        half = max(1, band_width_y // 2)
        draw.rectangle((0, seam - half, width, seam + half), fill=(0, 0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)


def composite_seam_repair(
    base_path: Path,
    repair_path: Path,
    mask_path: Path,
    output_path: Path,
    feather: int,
) -> None:
    with Image.open(base_path) as opened:
        base = opened.convert("RGB")
    with Image.open(repair_path) as opened:
        repair = opened.convert("RGB")
    if repair.width < base.width or repair.height < base.height:
        raise SystemExit(
            f"Seam repair returned {repair.width}x{repair.height}, below the "
            f"{base.width}x{base.height} canvas. Refusing to upscale the repair."
        )
    if repair.size != base.size:
        repair = repair.resize(base.size, Image.Resampling.LANCZOS)
    with Image.open(mask_path) as opened:
        mask_alpha = opened.getchannel("A")
    selector = Image.eval(mask_alpha, lambda alpha: 255 - alpha)
    if feather:
        selector = selector.filter(ImageFilter.GaussianBlur(feather))
    result = Image.composite(repair, base, selector)
    result.save(output_path)


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
    raise SystemExit(
        "codex-gpt-image CLI not found. Install the codex-gpt-image skill or pass --gpt-cli."
    )


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if not environment.get("SSL_CERT_FILE"):
        try:
            import certifi

            environment["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass
    return environment


def run_command(
    command: list[str], environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(message)
    return result


def read_user_instruction(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.prompt_file:
        prompt_path = args.prompt_file.expanduser().resolve()
        if not prompt_path.is_file():
            raise SystemExit(f"Prompt file not found: {prompt_path}")
        parts.append(prompt_path.read_text(encoding="utf-8").strip())
    if args.instruction:
        parts.append(args.instruction.strip())
    return "\n".join(part for part in parts if part)


def tile_prompt(
    tile: dict,
    canvas_width: int,
    canvas_height: int,
    user_instruction: str,
    context_image: bool,
) -> str:
    role_line = (
        "Input Image 1 is the local tile to redraw. Input Image 2 is a complete "
        "composition anchor and is context only. Output the local tile, never the "
        "complete composition."
        if context_image
        else "The input image is the tile to redraw."
    )
    guidance = (
        f"\nUser-specific guidance:\n{user_instruction}\n"
        if user_instruction
        else ""
    )
    return f"""Asset type: high-resolution overlapping tile reconstruction.
{role_line}
Tile location: row {tile['row']}, column {tile['column']}; pixel origin
({tile['x']}, {tile['y']}) on a {canvas_width}x{canvas_height} target canvas.

Redraw only the tile image as a faithful, sharper, higher-detail version of the
same crop. Preserve the exact composition, crop, perspective, geometry, object
positions, proportions, colors, lighting, visible text, numbers, logos, faces,
product shapes, and all content touching every edge. This tile overlaps adjacent
tiles, so boundary content must remain aligned. Do not zoom, shift, rotate,
reframe, add, remove, replace, reinterpret, or invent objects. Do not add a
watermark or new text. Return one image with the same aspect ratio as the tile.
{guidance}"""


def normalize_tile(
    path: Path,
    expected_size: tuple[int, int],
    alpha_source: Path | None = None,
) -> tuple[bool, tuple[int, int], float]:
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
    native_size = image.size
    upscale_factor = max(
        expected_size[0] / native_size[0],
        expected_size[1] / native_size[1],
    )
    if native_size[0] < expected_size[0] or native_size[1] < expected_size[1]:
        raise SystemExit(
            f"GPT tile returned {native_size[0]}x{native_size[1]}, but the tile "
            f"needs {expected_size[0]}x{expected_size[1]}. Refusing all pixel "
            "enlargement. Use smaller tiles so native redraws cover them."
        )
    if image.size == expected_size:
        if alpha_source:
            with Image.open(alpha_source) as opened:
                alpha = opened.convert("RGBA").getchannel("A")
            if alpha.size != expected_size:
                alpha = alpha.resize(expected_size, Image.Resampling.LANCZOS)
            if alpha.getextrema() != (255, 255):
                image.putalpha(alpha)
        image.save(path)
        return False, native_size, upscale_factor
    image = image.resize(expected_size, Image.Resampling.LANCZOS)
    if alpha_source:
        with Image.open(alpha_source) as opened:
            alpha = opened.convert("RGBA").getchannel("A")
        if alpha.size != expected_size:
            alpha = alpha.resize(expected_size, Image.Resampling.LANCZOS)
        if alpha.getextrema() != (255, 255):
            image.putalpha(alpha)
    image.save(path)
    return True, native_size, upscale_factor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resize an image to a target canvas, redraw overlapping tiles through "
            "codex-gpt-image, and merge the results."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--content-mode",
        required=True,
        choices=(
            "texture",
            "background",
            "scenery",
            "abstract",
            "product",
            "people",
            "text",
            "architecture",
            "ui",
            "diagram",
            "general",
        ),
        help="Semantic risk class. Structural modes are blocked by default.",
    )
    parser.add_argument(
        "--allow-structural-tiling",
        action="store_true",
        help=(
            "Allow a structural job only when --context-image supplies a global "
            "coherence anchor. Prefer gpt_coherent_redraw.py."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        help="Scale from source dimensions. Omit for the default 3840x2160 canvas.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        help="Named output size. Default: 4k-landscape.",
    )
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--tile-size", type=int, help="Legacy square tile shortcut.")
    parser.add_argument("--tile-width", type=int)
    parser.add_argument("--tile-height", type=int)
    parser.add_argument("--overlap", type=int, help="Legacy equal-overlap shortcut.")
    parser.add_argument("--overlap-x", type=int)
    parser.add_argument("--overlap-y", type=int)
    parser.add_argument(
        "--merge-mode",
        choices=("core", "feather"),
        help=(
            "Default: feather for low-risk scenery/texture/background/abstract; "
            "core for structural content."
        ),
    )
    parser.add_argument(
        "--skip-seam-repair",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seam-repair",
        dest="skip_seam_repair",
        action="store_false",
        help=(
            "Run an optional full-canvas masked repair. Disabled by default because "
            "a sub-target native response is rejected instead of being enlarged."
        ),
    )
    parser.add_argument(
        "--seam-feather",
        type=int,
        default=24,
        help="Pixel feather applied when reinserting repaired seam bands.",
    )
    parser.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="high")
    parser.add_argument("--instruction")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--context-image",
        type=Path,
        help="Complete globally coherent composition used only as a tile anchor.",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--gpt-cli")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tiles", type=int, default=64)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument("--allow-aspect-change", action="store_true")
    parser.add_argument(
        "--no-color-harmonization",
        action="store_true",
        help="Skip source-anchored low-frequency luminance/chroma matching.",
    )
    parser.add_argument("--color-blur-radius", type=float, default=96)
    parser.add_argument("--color-luma-limit", type=float, default=28)
    parser.add_argument("--color-chroma-limit", type=float, default=18)
    parser.set_defaults(skip_seam_repair=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    structural_modes = {
        "product",
        "people",
        "text",
        "architecture",
        "ui",
        "diagram",
        "general",
    }
    if args.content_mode in structural_modes and not args.allow_structural_tiling:
        raise SystemExit(
            f"Generative tiling is unsafe for content mode '{args.content_mode}'. "
            "Use gpt_coherent_redraw.py to establish a global structure anchor "
            "before tile redraw."
        )
    if args.content_mode in structural_modes and not args.context_image:
        raise SystemExit(
            "Structural tiling requires --context-image. Use gpt_coherent_redraw.py "
            "to create and apply the anchor automatically."
        )
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input image not found: {source}")
    if output.exists() and not args.force_output:
        raise SystemExit(f"Output already exists: {output}. Pass --force-output to replace it.")
    if args.scale is not None and args.scale < 1:
        raise SystemExit("--scale must be at least 1.")
    if (args.target_width is None) != (args.target_height is None):
        raise SystemExit("Pass both --target-width and --target-height.")
    target_modes = sum(
        (
            args.scale is not None,
            args.preset is not None,
            args.target_width is not None,
        )
    )
    if target_modes > 1:
        raise SystemExit("Choose only one of --scale, --preset, or custom target dimensions.")
    if args.seam_feather < 0:
        raise SystemExit("--seam-feather must be non-negative.")
    if args.color_blur_radius <= 0:
        raise SystemExit("--color-blur-radius must be positive.")
    if args.color_luma_limit < 0 or args.color_chroma_limit < 0:
        raise SystemExit("Color correction limits must be non-negative.")

    with Image.open(source) as opened:
        source_width, source_height = opened.size
    if args.target_width is not None:
        target_width = args.target_width
        target_height = args.target_height
    elif args.scale is not None:
        target_width = source_width * args.scale
        target_height = source_height * args.scale
    else:
        target_width, target_height = PRESETS[args.preset or "4k-landscape"]
    if target_width <= 0 or target_height <= 0:
        raise SystemExit("Target dimensions must be positive.")

    (
        default_tile_width,
        default_tile_height,
        default_overlap_x,
        default_overlap_y,
    ) = default_tile_geometry(
        target_width,
        target_height,
        args.content_mode in structural_modes,
    )
    tile_width = args.tile_width or args.tile_size or default_tile_width
    tile_height = args.tile_height or args.tile_size or default_tile_height
    overlap_x = (
        args.overlap_x
        if args.overlap_x is not None
        else args.overlap if args.overlap is not None else default_overlap_x
    )
    overlap_y = (
        args.overlap_y
        if args.overlap_y is not None
        else args.overlap if args.overlap is not None else default_overlap_y
    )
    request_width, request_height = largest_model_size(tile_width, tile_height)
    tile_pixels = request_width * request_height
    tile_ratio = max(request_width, request_height) / min(
        request_width, request_height
    )
    if max(request_width, request_height) > 3840 or tile_ratio > 3:
        raise SystemExit("Tile dimensions exceed GPT Image size or aspect-ratio limits.")
    if not MIN_MODEL_PIXELS <= tile_pixels <= MAX_MODEL_PIXELS:
        raise SystemExit("Tile pixel count must be between 655,360 and 8,294,400.")
    if overlap_x < 0 or overlap_x >= tile_width:
        raise SystemExit("Horizontal overlap must be smaller than tile width.")
    if overlap_y < 0 or overlap_y >= tile_height:
        raise SystemExit("Vertical overlap must be smaller than tile height.")
    if args.max_tiles < 1:
        raise SystemExit("--max-tiles must be positive.")
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative.")
    context_image = (
        args.context_image.expanduser().resolve() if args.context_image else None
    )
    if context_image and not context_image.is_file():
        raise SystemExit(f"Context image not found: {context_image}")

    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    ratio_difference = abs(source_ratio - target_ratio) / source_ratio
    if ratio_difference > 0.01 and not args.allow_aspect_change:
        raise SystemExit(
            "Source and target aspect ratios differ by more than 1%. "
            "Crop or extend the source to the target ratio first, or pass "
            "--allow-aspect-change only if stretching is intentional."
        )
    if target_width < tile_width or target_height < tile_height:
        raise SystemExit(
            "Target dimensions must be at least the corresponding tile dimensions."
        )
    if (
        not args.skip_seam_repair
        and (target_width > 3840 or target_height > 3840)
    ):
        raise SystemExit(
            "Automatic full-canvas seam repair supports dimensions up to 3840 per side. "
            "Use a smaller custom size or pass --skip-seam-repair."
        )

    user_instruction = read_user_instruction(args)
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output.parent / f".{output.stem}-gpt-tiles"
    )
    source_tiles = work_dir / "source"
    processed_tiles = work_dir / "processed"
    manifest_path = source_tiles / "manifest.json"
    tile_script = Path(__file__).resolve().with_name("tile_image.py")
    if manifest_path.exists() and not args.resume:
        raise SystemExit(
            f"Work directory already contains a run: {work_dir}. "
            "Pass --resume or choose a new --work-dir."
        )
    if not manifest_path.exists():
        source_tiles.mkdir(parents=True, exist_ok=True)
        split_command = [
            sys.executable,
            str(tile_script),
            "split",
            str(source),
            str(source_tiles),
            "--target-width",
            str(target_width),
            "--target-height",
            str(target_height),
            "--tile-width",
            str(tile_width),
            "--tile-height",
            str(tile_height),
            "--overlap-x",
            str(overlap_x),
            "--overlap-y",
            str(overlap_y),
        ]
        try:
            split_result = run_command(split_command)
        except RuntimeError as exc:
            raise SystemExit(f"Tile preparation failed: {exc}") from exc
        if split_result.stdout.strip():
            print(split_result.stdout.strip())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiles = manifest["tiles"]
    if len(tiles) > args.max_tiles:
        raise SystemExit(
            f"This job needs {len(tiles)} GPT Image calls, above --max-tiles "
            f"{args.max_tiles}. Increase the limit explicitly after reviewing cost and time."
        )

    print(
        f"Plan: {source_width}x{source_height} -> {target_width}x{target_height}; "
        f"content={args.content_mode}; {len(tiles)} tiles; "
        f"tile={tile_width}x{tile_height}; "
        f"overlap={overlap_x}x{overlap_y}"
    )
    if (request_width, request_height) != (tile_width, tile_height):
        print(
            f"GPT request size: {request_width}x{request_height}; "
            "responses will be normalized to tile dimensions"
        )
    print(f"Work directory: {work_dir}")
    if args.dry_run:
        print("Dry run complete; no GPT Image requests were sent.")
        return 0

    gpt_cli = find_gpt_cli(args.gpt_cli)
    environment = subprocess_environment()
    try:
        auth_result = run_command(
            [sys.executable, str(gpt_cli), "auth-status"],
            environment,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Codex OAuth check failed: {exc}") from exc
    if auth_result.stdout.strip():
        print(auth_result.stdout.strip())

    processed_tiles.mkdir(parents=True, exist_ok=True)
    resized_outputs = 0
    previous_run_path = work_dir / "run.json"
    previous_native_by_file: dict[str, dict[str, object]] = {}
    if args.resume and previous_run_path.is_file():
        try:
            previous_run = json.loads(previous_run_path.read_text(encoding="utf-8"))
            previous_native_by_file = {
                str(item["file"]): item
                for item in previous_run.get("tile_native_dimensions", [])
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            previous_native_by_file = {}
    tile_native_dimensions: list[dict[str, object]] = []
    for index, tile in enumerate(tiles, start=1):
        source_tile = source_tiles / tile["file"]
        processed_tile = processed_tiles / tile["file"]
        if processed_tile.is_file() and args.resume:
            print(f"[{index}/{len(tiles)}] Resume: {processed_tile.name}", flush=True)
            if processed_tile.name in previous_native_by_file:
                tile_native_dimensions.append(
                    previous_native_by_file[processed_tile.name]
                )
            else:
                with Image.open(processed_tile) as resumed:
                    resumed_size = resumed.size
                tile_native_dimensions.append(
                    {
                        "file": processed_tile.name,
                        "native": None,
                        "normalized": list(resumed_size),
                        "normalization_upscale": None,
                        "native_dimensions_unavailable_after_resume": True,
                    }
                )
            continue

        prompt = tile_prompt(
            tile,
            target_width,
            target_height,
            user_instruction,
            context_image is not None,
        )
        command = [
            sys.executable,
            str(gpt_cli),
            "generate",
            "--prompt",
            prompt,
            "--image",
            str(source_tile),
        ]
        if context_image:
            command.extend(["--image", str(context_image)])
        command.extend(
            [
                "--size",
                f"{request_width}x{request_height}",
                "--quality",
                args.quality,
                "--output-format",
                "png",
                "--out",
                str(processed_tile),
                "--timeout",
                "300",
            ]
        )

        print(f"[{index}/{len(tiles)}] GPT redraw: {source_tile.name}", flush=True)
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
                    print(f"Retrying tile after error: {exc}", flush=True)
                    time.sleep(1)
        if error:
            raise SystemExit(
                f"Tile {source_tile.name} failed; rerun with --resume after fixing: {error}"
            )
        expected_size = (int(tile["width"]), int(tile["height"]))
        normalized, native_size, normalization_upscale = normalize_tile(
            processed_tile,
            expected_size,
            source_tile,
        )
        tile_native_dimensions.append(
            {
                "file": processed_tile.name,
                "native": list(native_size),
                "normalized": list(expected_size),
                "normalization_upscale": normalization_upscale,
            }
        )
        if normalized:
            resized_outputs += 1
            print(
                f"Normalized {processed_tile.name} to "
                f"{expected_size[0]}x{expected_size[1]}"
            )

    low_risk_modes = {"texture", "background", "scenery", "abstract"}
    merge_mode = args.merge_mode or (
        "feather" if args.content_mode in low_risk_modes else "core"
    )
    merge_tiles = processed_tiles
    color_harmonization = len(tiles) > 1 and not args.no_color_harmonization
    harmonization_report: Path | None = None
    if color_harmonization:
        harmonized_tiles = work_dir / "harmonized"
        harmonize_script = Path(__file__).resolve().with_name("harmonize_tiles.py")
        try:
            harmonize_result = run_command(
                [
                    sys.executable,
                    str(harmonize_script),
                    str(manifest_path),
                    str(source_tiles),
                    str(processed_tiles),
                    str(harmonized_tiles),
                    "--blur-radius",
                    str(args.color_blur_radius),
                    "--luma-limit",
                    str(args.color_luma_limit),
                    "--chroma-limit",
                    str(args.color_chroma_limit),
                ]
            )
        except RuntimeError as exc:
            raise SystemExit(f"Tile color harmonization failed: {exc}") from exc
        if harmonize_result.stdout.strip():
            print(harmonize_result.stdout.strip())
        merge_tiles = harmonized_tiles
        harmonization_report = harmonized_tiles / "harmonization.json"

    merge_command = [
        sys.executable,
        str(tile_script),
        "merge",
        str(manifest_path),
        str(merge_tiles),
        str(output),
        "--mode",
        merge_mode,
    ]
    if args.force_output:
        merge_command.append("--force")
    try:
        merge_result = run_command(merge_command)
    except RuntimeError as exc:
        raise SystemExit(f"Tile merge failed: {exc}") from exc
    if merge_result.stdout.strip():
        print(merge_result.stdout.strip())

    seam_repair_used = False
    seam_repair_raw_size: tuple[int, int] | None = None
    if len(tiles) > 1 and not args.skip_seam_repair:
        seam_mask = work_dir / "seam-mask.png"
        seam_repair_raw = work_dir / "seam-repair-raw.png"
        create_seam_mask(
            manifest,
            seam_mask,
            max(16, overlap_x),
            max(16, overlap_y),
        )
        if not (args.resume and seam_repair_raw.is_file()):
            seam_prompt = """Seam repair for a high-resolution reconstructed image.
Edit only the transparent masked seam bands. Make all colors, gradients, lighting,
textures, edges, lines, branches, product geometry, typography, and other content
continue naturally across every join. Remove straight seams, cutoffs, duplicates,
ghosts, color shifts, and exposure steps. Preserve every unmasked pixel and the
overall composition exactly. Do not add, remove, move, redesign, reframe, or invent
content. Do not add text or a watermark. Return the complete image."""
            repair_command = [
                sys.executable,
                str(gpt_cli),
                "generate",
                "--prompt",
                seam_prompt,
                "--image",
                str(output),
                "--mask",
                str(seam_mask),
                "--size",
                f"{target_width}x{target_height}",
                "--quality",
                args.quality,
                "--output-format",
                "png",
                "--out",
                str(seam_repair_raw),
                "--timeout",
                "300",
            ]
            print("Final GPT seam repair", flush=True)
            try:
                repair_result = run_command(repair_command, environment)
            except RuntimeError as exc:
                raise SystemExit(
                    f"Seam repair failed; rerun with --resume after fixing: {exc}"
                ) from exc
            if repair_result.stderr.strip():
                print(repair_result.stderr.strip())
        with Image.open(seam_repair_raw) as repaired:
            seam_repair_raw_size = repaired.size
        composite_seam_repair(
            output,
            seam_repair_raw,
            seam_mask,
            output,
            args.seam_feather,
        )
        seam_repair_used = True
        print(
            f"Reinserted repaired seams into {target_width}x{target_height} canvas; "
            f"raw repair={seam_repair_raw_size[0]}x{seam_repair_raw_size[1]}"
        )

    run_metadata = {
        "input": str(source),
        "output": str(output),
        "content_mode": args.content_mode,
        "structural_tiling_acknowledged": args.allow_structural_tiling,
        "source_dimensions": [source_width, source_height],
        "target_dimensions": [target_width, target_height],
        "tile_dimensions": [tile_width, tile_height],
        "gpt_request_dimensions": [request_width, request_height],
        "overlap": [overlap_x, overlap_y],
        "merge_mode": merge_mode,
        "preset": args.preset,
        "tile_count": len(tiles),
        "quality": args.quality,
        "global_context": str(context_image) if context_image else None,
        "tile_native_dimensions": tile_native_dimensions,
        "pixel_enlargement_allowed": False,
        "normalized_tile_outputs": resized_outputs,
        "color_harmonization": color_harmonization,
        "color_harmonization_report": (
            str(harmonization_report) if harmonization_report else None
        ),
        "seam_repair": seam_repair_used,
        "seam_feather": args.seam_feather,
        "seam_repair_raw_dimensions": seam_repair_raw_size,
        "model": "gpt-image-2",
        "authentication": "Codex OAuth",
    }
    (work_dir / "run.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Completed GPT tile redraw: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
