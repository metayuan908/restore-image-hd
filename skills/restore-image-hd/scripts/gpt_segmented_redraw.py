#!/usr/bin/env python3
"""Redraw semantic regions through GPT Image masks and composite them exactly."""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import sys
import time
from pathlib import Path

from redraw_runtime import (Journal, JobLock, add_runtime_arguments, atomic_json,
                            file_hash, native_budget, run_jobs, save_capacity,
                            scoped_profile, signature, utc_now)
from redraw_quality import (align_donor, alignment_valid_mask, inspect_native,
                            minimum_error_selector, normalize_native)

try:
    from PIL import Image, ImageChops, ImageFilter, ImageStat
except ImportError as exc:
    raise SystemExit("Pillow is required. Use the Codex bundled Python runtime.") from exc

from gpt_tile_redraw import (
    PRESETS,
    find_gpt_cli,
    largest_model_size,
    run_command,
    subprocess_environment,
)


def odd_filter_size(radius: int) -> int:
    return max(3, radius * 2 + 1)


def expand_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(odd_filter_size(radius)))


def erode_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.MinFilter(odd_filter_size(radius)))


def load_binary_mask(
    path: Path,
    size: tuple[int, int],
    *,
    transparent_is_selected: bool,
) -> Image.Image:
    with Image.open(path) as opened:
        if "A" in opened.getbands() and opened.getchannel("A").getextrema() != (
            255,
            255,
        ):
            alpha = opened.getchannel("A")
            mask = (
                Image.eval(alpha, lambda value: 255 - value)
                if transparent_is_selected
                else alpha
            )
        else:
            mask = opened.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda value: 255 if value >= 128 else 0)


def load_edit_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    return load_binary_mask(path, size, transparent_is_selected=True)


def load_protected_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    return load_binary_mask(path, size, transparent_is_selected=False)


def subtract_mask(edit: Image.Image, protected: Image.Image) -> Image.Image:
    return ImageChops.subtract(edit, protected)


def union_masks(masks: list[Image.Image], size: tuple[int, int]) -> Image.Image:
    result = Image.new("L", size, 0)
    for mask in masks:
        result = ImageChops.lighter(result, mask)
    return result


def padded_bbox(
    mask: Image.Image, padding_ratio: float, minimum_padding: int
) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("mask has no editable pixels")
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    pad_x = max(minimum_padding, round(width * padding_ratio))
    pad_y = max(minimum_padding, round(height * padding_ratio))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(mask.width, right + pad_x),
        min(mask.height, bottom + pad_y),
    )


def mask_for_gpt(edit_mask: Image.Image) -> Image.Image:
    alpha = Image.eval(edit_mask, lambda value: 255 - value)
    result = Image.new("RGBA", edit_mask.size, (255, 255, 255, 255))
    result.putalpha(alpha)
    return result


def request_bbox(bbox, canvas_size):
    """Expand context for narrow masks; never stretch a long thin edge ring."""
    left,top,right,bottom=bbox
    width,height=right-left,bottom-top
    need_width=max(width,math.ceil(height/3))
    need_height=max(height,math.ceil(width/3))
    if need_width>canvas_size[0] or need_height>canvas_size[1]:
        raise RuntimeError("Region cannot fit GPT's 3:1 aspect limit; split at a semantic boundary")
    left=max(0,min(canvas_size[0]-need_width,(left+right-need_width)//2))
    top=max(0,min(canvas_size[1]-need_height,(top+bottom-need_height)//2))
    return left,top,left+need_width,top+need_height


def normalize_donor(
    path: Path,
    expected: tuple[int, int],
    normalized_path: Path,
) -> tuple[tuple[int, int], Image.Image]:
    return normalize_native(path, expected, normalized_path)


def harmonize_donor(
    source: Image.Image,
    donor: Image.Image,
    blur_radius: float,
    luma_limit: float,
    chroma_limit: float,
) -> tuple[Image.Image, dict[str, object]]:
    """Match spatial low-frequency Y/Cb/Cr response without changing geometry."""
    source_channels = source.convert("YCbCr").split()
    donor_channels = donor.convert("YCbCr").split()
    corrected_channels = []
    mean_delta = []
    limits = (luma_limit, chroma_limit, chroma_limit)
    for source_channel, donor_channel, limit in zip(
        source_channels, donor_channels, limits
    ):
        source_low = source_channel.filter(ImageFilter.GaussianBlur(blur_radius))
        donor_low = donor_channel.filter(ImageFilter.GaussianBlur(blur_radius))
        centered_delta = ImageChops.subtract(
            source_low,
            donor_low,
            scale=1.0,
            offset=128,
        )
        centered_delta = centered_delta.point(
            lambda value, channel_limit=limit: int(
                max(128 - channel_limit, min(128 + channel_limit, value))
            )
        )
        mean_delta.append(ImageStat.Stat(centered_delta).mean[0] - 128)
        corrected_channels.append(
            ImageChops.add(
                donor_channel,
                centered_delta,
                scale=1.0,
                offset=-128,
            )
        )
    corrected = Image.merge("YCbCr", corrected_channels).convert("RGB")
    return corrected, {
        "blur_radius": blur_radius,
        "luma_limit": luma_limit,
        "chroma_limit": chroma_limit,
        "mean_applied_ycbcr": [round(value, 3) for value in mean_delta],
    }


def transfer_detail(
    source: Image.Image,
    donor: Image.Image,
    blur_radius: float,
    strength: float,
    detail_limit: float,
) -> tuple[Image.Image, dict[str, object]]:
    """Transfer GPT high-frequency luminance while locking source color and tone."""
    source_ycc = source.convert("YCbCr")
    donor_ycc = donor.convert("YCbCr")
    source_y, source_cb, source_cr = source_ycc.split()
    donor_y = donor_ycc.getchannel("Y")
    source_low = source_y.filter(ImageFilter.GaussianBlur(blur_radius))
    donor_low = donor_y.filter(ImageFilter.GaussianBlur(blur_radius))
    source_high = ImageChops.subtract(source_y, source_low, scale=1.0, offset=128)
    donor_high = ImageChops.subtract(donor_y, donor_low, scale=1.0, offset=128)
    residual = ImageChops.subtract(donor_high, source_high, scale=1.0, offset=128)
    residual = residual.point(
        lambda value: int(
            max(
                128 - detail_limit,
                min(
                    128 + detail_limit,
                    128 + (value - 128) * strength,
                ),
            )
        )
    )
    result_y = ImageChops.add(source_y, residual, scale=1.0, offset=-128)
    result = Image.merge("YCbCr", (result_y, source_cb, source_cr)).convert("RGB")
    return result, {
        "blur_radius": blur_radius,
        "strength": strength,
        "detail_limit": detail_limit,
        "mean_luma_residual": round(ImageStat.Stat(residual).mean[0] - 128, 3),
    }


def boundary_delta(
    base: Image.Image,
    donor: Image.Image,
    edit_mask: Image.Image,
    radius: int,
) -> float:
    outside = expand_mask(edit_mask, radius)
    inside = erode_mask(edit_mask, radius)
    ring = ImageChops.subtract(outside, inside)
    if ring.getbbox() is None:
        return 0.0
    difference = ImageChops.difference(base.convert("RGB"), donor.convert("RGB")).convert(
        "L"
    )
    return ImageStat.Stat(difference, mask=ring).mean[0] / 255.0


def composite_region(
    canvas: Image.Image,
    donor: Image.Image,
    edit_mask: Image.Image,
    bbox: tuple[int, int, int, int],
    feather: int,
) -> None:
    selector = edit_mask.crop(bbox)
    if feather > 0:
        selector = ImageChops.darker(selector.filter(ImageFilter.GaussianBlur(feather)), selector)
    base_crop = canvas.crop(bbox).convert("RGB")
    merged = Image.composite(donor.convert("RGB"), base_crop, selector)
    canvas.paste(merged, bbox[:2])


def build_prompt(
    region: dict,
    pass_name: str,
    has_context: bool,
    user_instruction: str,
) -> str:
    name = str(region["name"])
    mode = str(region.get("mode", "general"))
    instruction = str(region.get("instruction", "")).strip()
    context_line = (
        "Input Image 2 is the complete composition and is context only."
        if has_context
        else ""
    )
    if pass_name == "edge":
        task = (
            "Repair only the transparent edge-ring mask so silhouettes, contacts, "
            "occlusion, lighting, color, and texture join the surrounding image naturally."
        )
    else:
        task = (
            "Redraw only the transparent mask as a faithful high-detail reconstruction "
            "of the same semantic region."
        )
    extra = "\n".join(part for part in (instruction, user_instruction) if part)
    return f"""Mask-driven local redraw.
Region: {name}
Semantic mode: {mode}
Input Image 1 is the authoritative local crop. {context_line}
{task}

Preserve the crop, camera, object count, silhouette, perspective, proportions,
contacts, colors, lighting, and every unmasked pixel. Do not zoom, shift, reframe,
duplicate, remove, redesign, or invent objects. Do not generate pseudo-text,
letters, prices, logos, QR codes, or watermarks. Return the local crop only, with
the same aspect ratio.
{extra}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Redraw SAM/SAM2/manual mask regions at high resolution and composite "
            "only their editable pixels into an exact delivery canvas."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--context-image", type=Path)
    parser.add_argument("--protect-mask", action="append", type=Path, default=[])
    parser.add_argument("--context-padding", type=float, default=0.22)
    parser.add_argument("--minimum-padding", type=int, default=32)
    parser.add_argument("--mask-expand", type=int, default=8)
    parser.add_argument("--mask-feather", type=int, default=6)
    parser.add_argument("--boundary-ring", type=int, default=8)
    parser.add_argument("--max-boundary-delta", type=float, default=0.55)
    parser.add_argument(
        "--no-color-harmonization",
        action="store_true",
        help="Skip source-anchored low-frequency Y/Cb/Cr matching.",
    )
    parser.add_argument("--color-blur-radius", type=float, default=96)
    parser.add_argument("--color-luma-limit", type=float, default=28)
    parser.add_argument("--color-chroma-limit", type=float, default=18)
    parser.add_argument("--detail-blur-radius", type=float, default=2.5)
    parser.add_argument("--detail-strength", type=float, default=0.65)
    parser.add_argument("--detail-limit", type=float, default=48)
    parser.add_argument(
        "--experimental-detail-only",
        action="store_true",
        help=(
            "Allow source-anchored detail transfer. This is an experimental hybrid "
            "route and must not be reported as a genuine full redraw."
        ),
    )
    parser.add_argument(
        "--minimum-detail-strength",
        type=float,
        default=0.5,
        help="Reject weaker detail-only regions; default 0.5.",
    )
    parser.add_argument(
        "--allow-source-baseline",
        action="store_true",
        help=(
            "Allow an enlarged source baseline when direct GPT redraw covers less "
            "than 85%% of the canvas. The result must be reported as hybrid."
        ),
    )
    parser.add_argument("--edge-pass", action="store_true")
    parser.add_argument("--edge-ring", type=int, default=12)
    parser.add_argument("--pilot-count", type=int, default=2)
    parser.add_argument(
        "--approve-pilot",
        action="store_true",
        help="Continue beyond the pilot only after its preview has been inspected.",
    )
    parser.add_argument("--max-regions", type=int, default=24)
    parser.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="high")
    parser.add_argument("--instruction")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--gpt-cli")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=1, help="Bounded rate-limit retries; ambiguous network failures require resume.")
    parser.add_argument("--stage", choices=("all", "prepare", "generate", "compose"), default="all")
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument("--allow-aspect-change", action="store_true")
    add_runtime_arguments(parser)
    return parser


def resolve_target(
    args: argparse.Namespace, source_size: tuple[int, int]
) -> tuple[int, int]:
    if (args.target_width is None) != (args.target_height is None):
        raise SystemExit("Pass both --target-width and --target-height.")
    if args.target_width is not None:
        target = (args.target_width, args.target_height)
    elif args.preset:
        target = PRESETS[args.preset]
    else:
        target = source_size
    if target[0] <= 0 or target[1] <= 0:
        raise SystemExit("Target dimensions must be positive.")
    return target


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def call_gpt(
    gpt_cli: Path,
    environment: dict[str, str],
    crop_path: Path,
    mask_path: Path,
    donor_path: Path,
    prompt: str,
    request_size: tuple[int, int],
    quality: str,
    context_image: Path | None,
    retries: int,
) -> None:
    command = [
        sys.executable,
        str(gpt_cli),
        "generate",
        "--prompt",
        prompt,
        "--image",
        str(crop_path),
        "--mask",
        str(mask_path),
    ]
    if context_image:
        command.extend(["--image", str(context_image)])
    command.extend(
        [
            "--size",
            f"{request_size[0]}x{request_size[1]}",
            "--quality",
            quality,
            "--output-format",
            "png",
            "--background",
            "opaque",
            "--out",
            str(donor_path),
            "--timeout",
            "300",
        ]
    )
    error: RuntimeError | None = None
    for attempt in range(retries + 1):
        try:
            result = run_command(command, environment)
            if result.stderr.strip():
                print(result.stderr.strip())
            error = None
            break
        except RuntimeError as exc:
            error = exc
            if attempt < retries:
                print(f"Retrying masked redraw after error: {exc}", flush=True)
                time.sleep(1)
    if error:
        raise error


def main() -> int:
    wall_started = time.monotonic()
    args = build_parser().parse_args()
    if not 655_360 <= args.native_pixel_budget <= 8_294_400:
        raise SystemExit("--native-pixel-budget must be between 655360 and 8294400.")
    if args.max_shift < 0 or args.max_edge_distance < 0:
        raise SystemExit("Geometry limits must be non-negative.")
    if args.alignment == "affine":
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("Affine alignment needs OpenCV before generation; use translation or install OpenCV.") from exc
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input image not found: {source}")
    if not manifest_path.is_file():
        raise SystemExit(f"Region manifest not found: {manifest_path}")
    if output.exists() and not args.force_output and not args.resume:
        raise SystemExit(f"Output already exists: {output}; pass --force-output to replace it.")
    if not 0 <= args.context_padding <= 2:
        raise SystemExit("--context-padding must be between 0 and 2.")
    if not 0 <= args.max_boundary_delta <= 1:
        raise SystemExit("--max-boundary-delta must be between 0 and 1.")
    if args.color_blur_radius <= 0:
        raise SystemExit("--color-blur-radius must be positive.")
    if args.color_luma_limit < 0 or args.color_chroma_limit < 0:
        raise SystemExit("Color correction limits must be non-negative.")
    if args.detail_blur_radius <= 0:
        raise SystemExit("--detail-blur-radius must be positive.")
    if not 0 <= args.detail_strength <= 2:
        raise SystemExit("--detail-strength must be between 0 and 2.")
    if not 0 < args.minimum_detail_strength <= 2:
        raise SystemExit("--minimum-detail-strength must be above 0 and at most 2.")
    if not 0 <= args.detail_limit <= 127:
        raise SystemExit("--detail-limit must be between 0 and 127.")
    if min(
        args.minimum_padding,
        args.mask_expand,
        args.mask_feather,
        args.boundary_ring,
        args.edge_ring,
        args.pilot_count,
        args.retries,
    ) < 0:
        raise SystemExit("Padding, mask, pilot, and retry values must be non-negative.")

    with Image.open(source) as opened:
        source_rgba = opened.convert("RGBA")
        source_size = opened.size
        source_alpha = (
            opened.getchannel("A").copy()
            if "A" in opened.getbands()
            and opened.getchannel("A").getextrema() != (255, 255)
            else None
        )
    target_size = resolve_target(args, source_size)
    source_ratio = source_size[0] / source_size[1]
    target_ratio = target_size[0] / target_size[1]
    ratio_difference = abs(source_ratio - target_ratio) / source_ratio
    if ratio_difference > 0.01 and not args.allow_aspect_change:
        raise SystemExit(
            "Source and target aspect ratios differ by more than 1%. Crop or extend "
            "the source before segmented redraw."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    regions = manifest.get("regions")
    if not isinstance(regions, list) or not regions:
        raise SystemExit("Manifest must contain a non-empty regions array.")
    if len(regions) > args.max_regions:
        raise SystemExit(
            f"Manifest has {len(regions)} regions, above --max-regions {args.max_regions}."
        )
    for index, region in enumerate(regions):
        if not isinstance(region, dict) or not region.get("name") or not region.get("mask"):
            raise SystemExit(f"Region {index + 1} requires name and mask.")
        if bool(region.get("detail_only", False)):
            if not args.experimental_detail_only:
                raise SystemExit(
                    f"Region {region['name']} uses detail_only. This hybrid route is "
                    "disabled by default; pass --experimental-detail-only only for "
                    "an explicitly disclosed experiment."
                )
            strength = float(region.get("detail_strength", args.detail_strength))
            if strength < args.minimum_detail_strength:
                raise SystemExit(
                    f"Region {region['name']} detail_strength {strength:.3f} is below "
                    f"the quality floor {args.minimum_detail_strength:.3f}. Do not "
                    "silently fall back to enlarged source detail."
                )
        region["_order"] = index
    names = [str(region["name"]) for region in regions]
    if len(names) != len(set(names)):
        raise SystemExit("Region names must be unique.")
    regions.sort(key=lambda item: (int(item.get("priority", 0)), int(item["_order"])))

    if source_rgba.size != target_size:
        canvas = source_rgba.resize(target_size, Image.Resampling.LANCZOS).convert("RGB")
    else:
        canvas = source_rgba.convert("RGB")
    if source_alpha and source_alpha.size != target_size:
        source_alpha = source_alpha.resize(target_size, Image.Resampling.LANCZOS)

    protected_paths = list(args.protect_mask)
    for value in manifest.get("protected_masks", []):
        protected_paths.append(resolve_manifest_path(manifest_path, str(value)))
    protected_masks = []
    for path in protected_paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise SystemExit(f"Protected mask not found: {resolved}")
        protected_masks.append(load_protected_mask(resolved, target_size))
    protected = union_masks(protected_masks, target_size)

    context_image = (
        args.context_image.expanduser().resolve() if args.context_image else None
    )
    if context_image and not context_image.is_file():
        raise SystemExit(f"Context image not found: {context_image}")

    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output.parent / f".{output.stem}-segmented"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    gpt_cli = find_gpt_cli(args.gpt_cli)
    capacity_scope = scoped_profile(gpt_cli, args.quality)
    generation_scope = scoped_profile(gpt_cli, args.quality, include_auth=False)
    profile_path = args.native_profile or (work_dir / "native-profile.json")
    if profile_path.is_file():
        try:
            budget = native_budget(profile_path, capacity_scope, args.native_pixel_budget)
        except ValueError:
            if args.native_profile:
                raise
            budget = args.native_pixel_budget
            print("Cached capacity profile expired/changed; using conservative planning budget.")
    else:
        if args.native_profile:
            raise SystemExit(f"Native profile does not exist: {profile_path}")
        budget = args.native_pixel_budget
    prepared: list[dict] = []
    plan_regions: list[dict] = []
    for region in regions:
        mask_path = resolve_manifest_path(manifest_path, str(region["mask"]))
        if not mask_path.is_file():
            raise SystemExit(f"Region mask not found: {mask_path}")
        edit = load_edit_mask(mask_path, target_size)
        edit = expand_mask(edit, int(region.get("expand", args.mask_expand)))
        edit = subtract_mask(edit, protected)
        try:
            bbox = padded_bbox(
                edit,
                float(region.get("context_padding", args.context_padding)),
                int(region.get("minimum_padding", args.minimum_padding)),
            )
        except ValueError as exc:
            raise SystemExit(f"Region {region['name']}: {exc}") from exc
        if region.get("crop_bbox") is not None:
            supplied = tuple(map(int, region["crop_bbox"]))
            core = edit.getbbox()
            if len(supplied) != 4 or not (0 <= supplied[0] <= core[0] < core[2] <= supplied[2] <= target_size[0]
                                       and 0 <= supplied[1] <= core[1] < core[3] <= supplied[3] <= target_size[1]):
                raise SystemExit(f"Region {region['name']}: crop_bbox must contain its editable mask within canvas.")
            bbox = supplied
        bbox = request_bbox(bbox,target_size)
        crop_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        if crop_size[0] > 3840 or crop_size[1] > 3840:
            raise SystemExit(
                f"Region {region['name']} crop {crop_size[0]}x{crop_size[1]} exceeds "
                "the 3840 edge limit. Split the semantic mask without cutting a "
                "connected subject, or reduce context padding."
            )
        if crop_size[0] * crop_size[1] > budget:
            raise SystemExit(f"Region {region['name']} crop {crop_size} exceeds measured/conservative native budget {budget} pixels including context. Replan smaller semantic regions before any image request.")
        request_size = largest_model_size(*crop_size, pixel_budget=max(1_500_000, budget))
        if request_size[0] < crop_size[0] or request_size[1] < crop_size[1]:
            raise SystemExit(
                f"Region {region['name']} needs crop {crop_size[0]}x{crop_size[1]}, "
                f"but the largest valid GPT request is {request_size[0]}x{request_size[1]}. "
                "Reduce the region crop; pixel enlargement is forbidden."
            )
        prepared.append(
            {
                "region": region,
                "edit": edit,
                "bbox": bbox,
                "crop_size": crop_size,
                "request_size": request_size,
            }
        )
        plan_regions.append(
            {
                "name": region["name"],
                "mode": region.get("mode", "general"),
                "mask": str(mask_path),
                "bbox": list(bbox),
                "crop_dimensions": list(crop_size),
                "gpt_request_dimensions": list(request_size),
                "pilot": bool(region.get("pilot")),
                "priority": int(region.get("priority", 0)),
            }
        )

    all_edit_masks = [item["edit"] for item in prepared]
    direct_redraw_masks = [
        item["edit"]
        for item in prepared
        if not bool(item["region"].get("detail_only", False))
    ]
    masked_coverage = (
        ImageStat.Stat(union_masks(all_edit_masks, target_size)).mean[0] / 255
    )
    full_redraw_coverage = (
        ImageStat.Stat(union_masks(direct_redraw_masks, target_size)).mean[0] / 255
        if direct_redraw_masks
        else 0.0
    )
    source_is_enlarged = (
        target_size[0] > source_size[0] or target_size[1] > source_size[1]
    )
    if (
        source_is_enlarged
        and full_redraw_coverage < 0.85
        and not args.allow_source_baseline
    ):
        raise SystemExit(
            "Direct GPT redraw covers only "
            f"{full_redraw_coverage:.1%} of the enlarged canvas. Refusing a "
            "dimension-only 4K result. Expand direct semantic redraw coverage to "
            "at least 85%, use the coherent redraw route, or pass "
            "--allow-source-baseline for an explicitly disclosed hybrid."
        )
    if any(bool(item["region"].get("detail_only", False)) for item in prepared):
        resolution_claim = "experimental-source-anchored-detail-transfer"
    elif source_is_enlarged and full_redraw_coverage < 0.85:
        resolution_claim = "hybrid-source-baseline"
    else:
        resolution_claim = "gpt-redrawn-delivery-canvas"

    plan = {
        "input": str(source),
        "output": str(output),
        "manifest": str(manifest_path),
        "source_dimensions": list(source_size),
        "target_dimensions": list(target_size),
        "region_count": len(prepared),
        "pilot_count": min(args.pilot_count, len(prepared)),
        "protected_mask_count": len(protected_masks),
        "edge_pass": args.edge_pass,
        "pixel_enlargement_allowed": False,
        "masked_coverage": round(masked_coverage, 6),
        "direct_gpt_redraw_coverage": round(full_redraw_coverage, 6),
        "source_baseline_allowed": args.allow_source_baseline,
        "resolution_claim": resolution_claim,
        "regions": plan_regions,
    }
    print(
        f"Plan: {source_size[0]}x{source_size[1]} -> "
        f"{target_size[0]}x{target_size[1]}; {len(prepared)} masked regions; "
        f"pilot={min(args.pilot_count, len(prepared))}; edge-pass={args.edge_pass}"
    )
    print(f"Work directory: {work_dir}")
    if args.dry_run:
        atomic_json(work_dir / "plan.json",plan)
        print("Dry run complete; no GPT Image requests were sent.")
        return 0

    # Source canvas is immutable throughout generation. Only this coordinator
    # mutates the final composition, in manifest priority order.
    baseline = canvas.copy()
    environment = subprocess_environment()
    journal = Journal(work_dir)
    journal.event("preflight_complete", seconds=time.monotonic()-wall_started,
                  native_pixel_budget=budget, workers=args.jobs)
    jobs = []
    item_by_id = {}
    generation_dispatch_seconds = 0.0
    context_digest = file_hash(context_image) if context_image else None
    processing_signature = signature({key: getattr(args, key) for key in (
        "alignment", "max_shift", "max_edge_distance", "seam_mode", "mask_feather",
        "boundary_ring", "max_boundary_delta", "no_color_harmonization",
        "color_blur_radius", "color_luma_limit", "color_chroma_limit",
        "detail_blur_radius", "detail_strength", "detail_limit")})

    def prepare_jobs(items, anchor, pass_name):
        result = []
        for index, item in enumerate(items):
            region = item["region"]
            safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(region["name"]))
            job_id = f"{index+1:03d}-{safe}-{pass_name}"
            prompt = build_prompt(region, pass_name, context_image is not None, args.instruction or "")
            crop = anchor.crop(item["bbox"]).convert("RGB")
            selector = mask_for_gpt(item["edit"].crop(item["bbox"]))
            fingerprint = signature({"schema": 2, "crop": hashlib.sha256(crop.tobytes()).hexdigest(),
                                     "crop_size": crop.size, "mask": hashlib.sha256(selector.tobytes()).hexdigest(),
                                     "context": context_digest, "prompt": prompt,
                                     "request_size": item["request_size"], "scope": generation_scope,
                                     "background": "opaque"})
            folder = work_dir / "inputs" / job_id / fingerprint
            folder.mkdir(parents=True, exist_ok=True)
            crop_path, mask_path = folder / "crop.png", folder / "mask.png"
            # Write exact deterministic inputs once; workers receive immutable paths.
            if not crop_path.is_file():
                crop.save(crop_path)
            if not mask_path.is_file():
                selector.save(mask_path)
            with Image.open(crop_path) as existing_crop, Image.open(mask_path) as existing_mask:
                if existing_crop.convert("RGB").tobytes() != crop.tobytes() or existing_mask.tobytes() != selector.tobytes():
                    raise RuntimeError(f"{job_id}: prepared inputs were changed")
            job = {"id": job_id, "signature": fingerprint, "prompt": prompt,
                   "crop_path": crop_path, "mask_path": mask_path,
                   "expected": item["crop_size"], "request_size": item["request_size"]}
            item_by_id[job_id] = item
            result.append(job)
        return result

    def generate(job, raw):
        print(f"GPT redraw: {job['id']}", flush=True)
        # Retries and scheduling are owned by the durable dispatcher, not nested loops.
        call_gpt(gpt_cli, environment, job["crop_path"], job["mask_path"], raw,
                 job["prompt"], job["request_size"], args.quality, context_image, 0)

    def inspect(raw, job):
        return inspect_native(raw, job["expected"])

    def dispatch(selected, **options):
        nonlocal generation_dispatch_seconds
        started = time.monotonic()
        try:
            resets={job["id"] for job in selected if job["id"] in args.retry_region or item_by_id[job["id"]]["region"]["name"] in args.retry_region}
            return run_jobs(selected, generate, inspect, work_dir, retries=args.retries,reset_ids=resets, **options)
        except RuntimeError:
            # An undersized return is still evidence about actual capacity.
            observed = []
            for job in selected:
                record = work_dir / "jobs" / job["id"] / job["signature"] / "record.json"
                if record.is_file():
                    row = json.loads(record.read_text())
                    if row.get("native_dimensions"):
                        observed.append(row)
            save_capacity(work_dir / "native-profile.json", observed, capacity_scope)
            raise
        finally:
            generation_dispatch_seconds += time.monotonic()-started

    def process(job, raw_record):
        begin = time.monotonic()
        item = item_by_id[job["id"]]
        folder = Path(raw_record["raw_path"]).parent
        processed = folder / f"processed-{processing_signature}.png"
        acceptance = folder / f"acceptance-{processing_signature}.json"
        digest = signature({"processing": processing_signature, "region_config": item["region"], "raw": raw_record["raw_sha256"],
                            "job": job["signature"]})
        if acceptance.is_file():
            cached = json.loads(acceptance.read_text())
            if cached.get("signature") == digest and processed.is_file() and file_hash(processed) == cached.get("processed_sha256"):
                cached.update(raw_record)
                cached.update(signature=digest, state="accepted",accepted=True,
                              resumed=raw_record.get("resumed",False),processing_resumed=True)
                journal.event("quality_resume",job=job["id"],seconds=0)
                with Image.open(processed) as opened:
                    return opened.convert("RGB"), cached
        native, donor = normalize_donor(Path(raw_record["raw_path"]), item["crop_size"], folder / "normalized.png")
        with Image.open(job["crop_path"]) as opened:
            base_crop = opened.convert("RGB")
        geometry_begin = time.monotonic()
        donor, geometry = align_donor(base_crop, donor, args.alignment, args.max_shift, args.max_edge_distance)
        valid = alignment_valid_mask(donor.size, geometry)
        edit_crop = item["edit"].crop(item["bbox"])
        # A mask touching an invalid mapped edge is rejected, not filled with black
        # or silently replaced by an enlarged source crop.
        lost = ImageChops.subtract(edit_crop, valid)
        if lost.getbbox() is not None:
            raise RuntimeError(f"{job['id']}: alignment needs missing native border pixels; use more context or regenerate")
        geometry_seconds = time.monotonic()-geometry_begin
        color_report = detail_report = None
        if bool(item["region"].get("detail_only", False)):
            donor, detail_report = transfer_detail(base_crop, donor,
                float(item["region"].get("detail_blur_radius", args.detail_blur_radius)),
                float(item["region"].get("detail_strength", args.detail_strength)),
                float(item["region"].get("detail_limit", args.detail_limit)))
        elif not args.no_color_harmonization:
            donor, color_report = harmonize_donor(base_crop, donor, args.color_blur_radius,
                                                  args.color_luma_limit, args.color_chroma_limit)
        delta = boundary_delta(base_crop, donor, edit_crop, int(item["region"].get("boundary_ring",args.boundary_ring)))
        if delta > args.max_boundary_delta:
            raise RuntimeError(f"{job['id']}: boundary delta {delta:.3f} exceeds {args.max_boundary_delta:.3f}")
        donor.save(processed)
        row = {**raw_record, "name": str(item["region"]["name"]), "pass": item.get("pass", "interior"),
               "signature": digest, "generation_signature": job["signature"],
               "native_dimensions": list(native), "normalized_dimensions": list(item["crop_size"]),
               "raw_donor": raw_record["raw_path"], "normalized_donor": str(folder / "normalized.png"),
               "processed_path": str(processed), "processed_sha256": file_hash(processed),
               "state": "accepted", "accepted": True, "boundary_delta": delta,
               "geometry": geometry, "geometry_seconds": geometry_seconds,
               "processing_seconds": time.monotonic()-begin, "color_harmonization": color_report,
               "detail_transfer": detail_report}
        atomic_json(acceptance, row)
        journal.event("quality_complete", job=job["id"], seconds=row["processing_seconds"],
                      edge_distance=geometry["after"]["p95_pixels"])
        return donor, row

    def compose_items(selected_jobs, raw_records, target):
        rows, coverage = [], Image.new("L", target_size, 0)
        failures = []
        for job, record in zip(selected_jobs, raw_records):
            item = item_by_id[job["id"]]
            try:
                donor, row = process(job, record)
            except RuntimeError as error:
                failures.append({"job":job["id"],"bbox":item["bbox"],"reason":str(error),"state":"quality_failed"})
                journal.event("quality_failed", **failures[-1])
                continue
            edit = item["edit"]
            box = item["bbox"]
            selector = edit.crop(box)
            seam = {"used": False, "reason": "feather mode"}
            if args.seam_mode == "minimum-error":
                selector, seam = minimum_error_selector(target.crop(box), donor, selector, coverage.crop(box))
            feather = int(item["region"].get("edge_feather" if item.get("pass")=="edge" else "feather",args.mask_feather))
            if feather:
                selector = ImageChops.darker(selector.filter(ImageFilter.GaussianBlur(feather)), edit.crop(box))
            target.paste(Image.composite(donor, target.crop(box), selector), box[:2])
            coverage = ImageChops.lighter(coverage, edit)
            rows.append({**row, "seam": seam})
            atomic_json(work_dir / "composition-checkpoint.json", {"state": "partial", "records": rows})
        atomic_json(work_dir / "quality-report.json", {"passed":len(rows),"failures":failures})
        if failures:
            raise RuntimeError(f"{len(failures)} region(s) failed quality checks; see quality-report.json. No final output was exported; regenerate only failed regions.")
        return rows

    with JobLock(work_dir):
        if context_image:
            context_snapshot = work_dir / "inputs" / ("context-"+context_digest+context_image.suffix)
            context_snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not context_snapshot.is_file():
                import shutil
                shutil.copyfile(context_image,context_snapshot)
            if file_hash(context_snapshot)!=context_digest:
                raise RuntimeError("Context input changed; cannot use an inconsistent global anchor")
            context_image = context_snapshot
        atomic_json(work_dir / "status.json", {"state":"preparing", "started_at":utc_now()})
        atomic_json(work_dir / "plan.json",plan)
        jobs = prepare_jobs(prepared, baseline, "interior")
        valid_region_ids={job["id"] for job in jobs}|{str(item["region"]["name"]) for item in prepared}
        if set(args.retry_region)-valid_region_ids:
            raise RuntimeError("Unknown --retry-region; use a name/job ID from prepared-jobs.json")
        atomic_json(work_dir / "prepared-jobs.json", [
            {key: str(value) if isinstance(value, Path) else value for key,value in job.items()} for job in jobs])
        if args.stage == "prepare":
            atomic_json(work_dir / "status.json", {"state":"prepared"})
            print("Immutable crops and masks prepared; no image requests sent.")
            return 0
        if args.stage != "compose":
            try:
                run_command([sys.executable, str(gpt_cli), "auth-status"], environment)
            except RuntimeError as exc:
                raise SystemExit(f"Codex OAuth check failed: {exc}") from exc
        else:
            def generate(job, raw):
                raise RuntimeError("Compose-only stage cannot generate; required verified donor is missing")
        flagged = [job for job in jobs if item_by_id[job["id"]]["region"].get("pilot")]
        pilot_jobs = (flagged or jobs)[:min(args.pilot_count,len(jobs))]
        pilot_ids = {job["id"] for job in pilot_jobs}
        pilot_input_signature = signature({"jobs": [job["signature"] for job in pilot_jobs],
                                     "regions": [item_by_id[job["id"]]["region"] for job in pilot_jobs],
                                     "processing": processing_signature})
        prior_pilot_path = work_dir / "pilot-review.json"
        prior_pilot = json.loads(prior_pilot_path.read_text()) if prior_pilot_path.is_file() else {}
        effective_resume = args.resume or args.stage == "compose"
        generation_begin = time.monotonic()
        pilot_records = dispatch(pilot_jobs, workers=min(args.jobs,2), resume=effective_resume)
        pilot_signature=signature({"inputs":pilot_input_signature,"raw":[row["raw_sha256"] for row in pilot_records]})
        previously_reviewable = prior_pilot.get("signature") == pilot_signature
        pilot_approved = previously_reviewable and (args.approve_pilot or prior_pilot.get("state")=="approved")
        if pilot_jobs:
            preview = baseline.copy()
            compose_items(pilot_jobs, pilot_records, preview)
            if source_alpha is not None:
                preview=preview.convert("RGBA")
                preview.putalpha(source_alpha)
            preview.save(work_dir / "pilot-composite.png")
            save_capacity(work_dir / "native-profile.json", pilot_records, capacity_scope)
            atomic_json(prior_pilot_path, {"signature":pilot_signature, "state":"approved" if pilot_approved else "ready_for_visual_review",
                                          "jobs":[job["id"] for job in pilot_jobs]})
            if len(pilot_jobs) < len(jobs) and not pilot_approved:
                atomic_json(work_dir / "status.json", {"state":"pilot_review", "signature":pilot_signature})
                print(f"Pilot ready: {work_dir / 'pilot-composite.png'}. Inspect at 100%, then --resume --approve-pilot. Remaining requests were not sent.")
                return 0
        remaining = [job for job in jobs if job["id"] not in pilot_ids]
        other_records = dispatch(remaining, workers=args.jobs, resume=effective_resume)
        by_id = {row["job"]:row for row in pilot_records+other_records}
        raw_records = [by_id[job["id"]] for job in jobs]
        generation_wall = time.monotonic()-generation_begin
        save_capacity(work_dir / "native-profile.json", raw_records, capacity_scope)
        if args.stage == "generate":
            atomic_json(work_dir / "status.json", {"state":"generated"})
            print("Generation checkpoint complete. Use --stage compose --resume for offline assembly.")
            return 0
        compose_begin = time.monotonic()
        generation_before_composition = generation_dispatch_seconds
        records = compose_items(jobs, raw_records, canvas)
        if args.edge_pass:
            # Freeze the entire interior composite before preparing edge jobs.
            edge_anchor = canvas.copy()
            edge_items = []
            for item in prepared:
                ring_width = int(item["region"].get("edge_ring", args.edge_ring))
                ring = subtract_mask(ImageChops.subtract(expand_mask(item["edit"],ring_width),
                                                        erode_mask(item["edit"],ring_width)),protected)
                if ring.getbbox() is not None:
                    # Recompute context for the expanded ring; never clip it to
                    # the old interior bbox (which can silently drop ring pixels).
                    box = request_bbox(padded_bbox(ring,args.context_padding,args.minimum_padding),target_size)
                    size = (box[2]-box[0],box[3]-box[1])
                    if size[0]*size[1] > budget:
                        raise RuntimeError("Edge crop exceeds native budget; prepare a focused repair manifest instead")
                    edge_items.append({**item,"edit":ring,"bbox":box,"crop_size":size,
                                       "request_size":largest_model_size(*size,pixel_budget=max(1_500_000,budget)),"pass":"edge"})
            edge_jobs = prepare_jobs(edge_items,edge_anchor,"edge")
            edge_records = dispatch(edge_jobs, workers=args.jobs, resume=effective_resume)
            records += compose_items(edge_jobs,edge_records,canvas)
        output.parent.mkdir(parents=True, exist_ok=True)
        if source_alpha is not None:
            result = canvas.convert("RGBA")
            result.putalpha(source_alpha)
        else:
            result = canvas
        temporary_output = work_dir / "verified-output.png"
        result.save(temporary_output)
        with Image.open(temporary_output) as checked:
            checked.load()
            if checked.size != target_size:
                raise RuntimeError("Final dimensions changed during export")
        import shutil
        shutil.copyfile(temporary_output, output)
        run = {**plan, "schema_version":2, "quality":args.quality,"records":records,
               "model":"gpt-image-2","authentication":"Codex OAuth",
               "source_sha256":file_hash(source),"output_sha256":file_hash(output),
               "native_pixel_budget":budget,"workers_requested":args.jobs,
               "immutable_generation_inputs":True,"composition_order":"manifest priority, then original order",
               "alignment":args.alignment,"seam_mode":args.seam_mode,
               "generation_wall_seconds":generation_dispatch_seconds,
               "generation_and_pilot_wall_seconds":generation_wall,
               "composition_wall_seconds":time.monotonic()-compose_begin-(generation_dispatch_seconds-generation_before_composition),
               "wall_seconds":time.monotonic()-wall_started,"finished_at":utc_now(),
               "detail_upscaler_used":False,"pipeline":"immutable masked crops -> bounded generation queue -> geometry QC -> color -> ordered seam composite"}
        atomic_json(work_dir / "run.json",run)
        atomic_json(work_dir / "status.json", {"state":"completed", "output_sha256":run["output_sha256"]})
        journal.event("delivery_complete", output=str(output),wall_seconds=run["wall_seconds"])
        print(f"Completed segmented GPT redraw: {output}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
