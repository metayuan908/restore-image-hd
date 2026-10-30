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


def largest_model_size(width: int, height: int, pixel_budget: int = 1_500_000) -> tuple[int, int]:
    ratio = width / height
    best: tuple[int, int, int, float] | None = None
    for request_width in range(16, MAX_MODEL_EDGE + 1, 16):
        request_height = round((request_width / ratio) / 16) * 16
        if request_height < 16 or request_height > MAX_MODEL_EDGE:
            continue
        pixels = request_width * request_height
        if not MIN_MODEL_PIXELS <= pixels <= min(MAX_MODEL_PIXELS, pixel_budget):
            continue
        ratio_error = abs(request_width / request_height - ratio) / ratio
        if ratio_error > 0.01 or max(request_width, request_height) > 3 * min(request_width, request_height):
            continue
        candidate = (pixels, request_width, request_height, ratio_error)
        if best is None or candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[3] < best[3]
        ):
            best = candidate
    if best is None:
        raise SystemExit("Cannot choose a valid GPT Image request size for this tile.")
    return best[1], best[2]


def default_tile_geometry(
    target_width: int, target_height: int, structural: bool, native_pixel_budget: int = 1_200_000
) -> tuple[int, int, int, int]:
    # Near-square destination crops use the observed/conservative capacity well.
    # Tile dimensions INCLUDE all overlap/context pixels sent for redraw.
    budget = native_pixel_budget
    if target_width * target_height <= budget:
        return target_width, target_height, 0, 0
    tile_width = min(target_width, max(16, int((budget * 1.2) ** .5) // 16 * 16))
    tile_height = min(target_height, max(16, (budget // tile_width) // 16 * 16))
    overlap_x = 0 if tile_width == target_width else min(128, tile_width // 8)
    overlap_y = 0 if tile_height == target_height else min(128, tile_height // 8)
    return tile_width,tile_height,overlap_x,overlap_y


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
        choices=("core", "feather", "minimum-error"),
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
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--pilot-count", type=int, default=1)
    parser.add_argument("--approve-pilot", action="store_true")
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
    from redraw_runtime import add_runtime_arguments
    add_runtime_arguments(parser)
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
    if output.exists() and not args.force_output and not args.resume:
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

    from redraw_runtime import native_budget, scoped_profile
    budget = native_budget(args.native_profile, scoped_profile(find_gpt_cli(args.gpt_cli), args.quality), args.native_pixel_budget)
    (
        default_tile_width,
        default_tile_height,
        default_overlap_x,
        default_overlap_y,
    ) = default_tile_geometry(
        target_width,
        target_height,
        args.content_mode in structural_modes,
        budget,
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
    if tile_width * tile_height > budget:
        raise SystemExit("Tile including overlap exceeds actual/conservative native budget; use smaller tiles before any image request.")
    request_width, request_height = largest_model_size(tile_width, tile_height, max(1_500_000,budget))
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
    expected_geometry={"canvas_width":target_width,"canvas_height":target_height,
                       "tile_width":tile_width,"tile_height":tile_height,
                       "overlap_x":overlap_x,"overlap_y":overlap_y}
    if any(manifest.get(k)!=v for k,v in expected_geometry.items()):
        raise SystemExit("Resumed tile layout differs; choose a new work directory instead of reusing old coordinates.")
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

    # All routes share the same durable generation/QC engine. The tiler owns
    # geometry; the engine owns immutable crops, bounded workers and checkpoints.
    from PIL import ImageDraw
    from redraw_runtime import runtime_flags, atomic_json
    region_masks = work_dir / "region-masks"
    region_masks.mkdir(parents=True, exist_ok=True)
    semantic_regions = []
    x_seams = axis_seams(tiles,"x")
    y_seams = axis_seams(tiles,"y")
    merge_mode = args.merge_mode or "minimum-error"
    for i,tile in enumerate(tiles):
        x,y,w,h = (int(tile[k]) for k in ("x","y","width","height"))
        box=(x,y,x+w,y+h)
        edit_box=list(box)
        if merge_mode == "core":
            # Mid-overlap ownership, preserving the legacy explicit core option.
            for seam in x_seams:
                if x < seam < x+w:
                    if seam <= x+w/2:edit_box[0]=max(edit_box[0],seam)
                    else:edit_box[2]=min(edit_box[2],seam)
            for seam in y_seams:
                if y < seam < y+h:
                    if seam <= y+h/2:edit_box[1]=max(edit_box[1],seam)
                    else:edit_box[3]=min(edit_box[3],seam)
        mask_path = region_masks / f"tile-{i+1:04d}.png"
        mask=Image.new("L",(target_width,target_height),0)
        ImageDraw.Draw(mask).rectangle((edit_box[0],edit_box[1],edit_box[2]-1,edit_box[3]-1),fill=255)
        mask.save(mask_path)
        semantic_regions.append({"name":f"tile-{i+1:04d}","mask":str(mask_path),
                                 "crop_bbox":box,"expand":0,"priority":i,
                                 "mode":args.content_mode,
                                 "instruction":tile_prompt(tile,target_width,target_height,user_instruction,context_image is not None)})
    regions_path=work_dir / "regions.json"
    atomic_json(regions_path,{"regions":semantic_regions})
    engine=Path(__file__).with_name("gpt_segmented_redraw.py")
    command=[sys.executable,str(engine),str(source),str(output),"--manifest",str(regions_path),
             "--target-width",str(target_width),"--target-height",str(target_height),
             "--work-dir",str(work_dir / "engine"),"--gpt-cli",str(find_gpt_cli(args.gpt_cli)),
             "--quality",args.quality,"--mask-expand","0","--context-padding","0",
             "--minimum-padding","0","--max-regions",str(args.max_tiles),
             "--pilot-count",str(args.pilot_count),"--retries",str(args.retries),
             "--color-blur-radius",str(args.color_blur_radius),"--color-luma-limit",str(args.color_luma_limit),
             "--color-chroma-limit",str(args.color_chroma_limit),*runtime_flags(args)]
    if context_image:command.extend(["--context-image",str(context_image)])
    if merge_mode in ("core","feather"):
        command.extend(["--seam-mode","feather"])
    if merge_mode=="core":command.extend(["--mask-feather","0"])
    if args.no_color_harmonization:command.append("--no-color-harmonization")
    for flag in ("resume","force_output","approve_pilot","allow_aspect_change"):
        if getattr(args,flag,False):command.append("--"+flag.replace("_","-"))
    if not args.skip_seam_repair:
        raise SystemExit("Full-canvas repair is no longer dispatched blindly. Inspect seam QA and submit only failed regions to the segmented runner.")
    # Stream progress instead of capturing an entire multi-minute child run.
    completed=subprocess.run(command,env=subprocess_environment())
    if completed.returncode:
        raise SystemExit(completed.returncode)
    engine_run=work_dir / "engine/run.json"
    engine_status_path=work_dir / "engine/status.json"
    engine_status=json.loads(engine_status_path.read_text()) if engine_status_path.is_file() else {}
    atomic_json(work_dir / "status.json",engine_status)
    if not engine_run.is_file() or engine_status.get("state")!="completed":
        print("Pilot needs visual review; rerun --resume --approve-pilot after inspection.")
        return 0
    metadata=json.loads(engine_run.read_text())
    metadata.update(tile_count=len(tiles),tile_dimensions=[tile_width,tile_height],
                    gpt_request_dimensions=[request_width,request_height],
                    tile_native_dimensions=[{"file":r["name"],"native":r["native_dimensions"],
                                             "normalized":r["normalized_dimensions"]} for r in metadata["records"]],
                    merge_mode=merge_mode)
    atomic_json(work_dir / "run.json",metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
