#!/usr/bin/env python3
"""Redraw semantic regions through GPT Image masks and composite them exactly."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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


def normalize_donor(
    path: Path,
    expected: tuple[int, int],
    normalized_path: Path,
) -> tuple[tuple[int, int], Image.Image]:
    with Image.open(path) as opened:
        donor = opened.convert("RGB")
    native = donor.size
    if donor.width < expected[0] or donor.height < expected[1]:
        raise RuntimeError(
            f"GPT returned {donor.width}x{donor.height}, below editable crop "
            f"{expected[0]}x{expected[1]}; refusing pixel enlargement"
        )
    if donor.size != expected:
        donor = donor.resize(expected, Image.Resampling.LANCZOS)
    donor.save(normalized_path)
    return native, donor


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
        selector = selector.filter(ImageFilter.GaussianBlur(feather))
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
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument("--allow-aspect-change", action="store_true")
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
    args = build_parser().parse_args()
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
    regions.sort(
        key=lambda item: (
            0 if item.get("pilot") else 1,
            int(item.get("priority", 0)),
            int(item["_order"]),
        )
    )

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
        crop_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        if crop_size[0] > 3840 or crop_size[1] > 3840:
            raise SystemExit(
                f"Region {region['name']} crop {crop_size[0]}x{crop_size[1]} exceeds "
                "the 3840 edge limit. Split the semantic mask without cutting a "
                "connected subject, or reduce context padding."
            )
        request_size = largest_model_size(*crop_size)
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
    (work_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Plan: {source_size[0]}x{source_size[1]} -> "
        f"{target_size[0]}x{target_size[1]}; {len(prepared)} masked regions; "
        f"pilot={min(args.pilot_count, len(prepared))}; edge-pass={args.edge_pass}"
    )
    print(f"Work directory: {work_dir}")
    if args.dry_run:
        print("Dry run complete; no GPT Image requests were sent.")
        return 0

    gpt_cli = find_gpt_cli(args.gpt_cli)
    environment = subprocess_environment()
    try:
        auth_result = run_command(
            [sys.executable, str(gpt_cli), "auth-status"], environment
        )
    except RuntimeError as exc:
        raise SystemExit(f"Codex OAuth check failed: {exc}") from exc
    if auth_result.stdout.strip():
        print(auth_result.stdout.strip())

    records: list[dict] = []
    for index, item in enumerate(prepared, start=1):
        region = item["region"]
        name = str(region["name"])
        safe_name = "".join(character if character.isalnum() else "-" for character in name)
        bbox = item["bbox"]
        edit = item["edit"]
        crop_path = work_dir / f"{index:02d}-{safe_name}-crop.png"
        mask_path = work_dir / f"{index:02d}-{safe_name}-mask.png"
        donor_path = work_dir / f"{index:02d}-{safe_name}-raw.png"
        normalized_path = work_dir / f"{index:02d}-{safe_name}-normalized.png"
        canvas.crop(bbox).save(crop_path)
        mask_for_gpt(edit.crop(bbox)).save(mask_path)
        prompt = build_prompt(
            region,
            "interior",
            context_image is not None,
            args.instruction or "",
        )
        resumed = args.resume and donor_path.is_file()
        if not resumed:
            print(f"[{index}/{len(prepared)}] Masked redraw: {name}", flush=True)
            try:
                call_gpt(
                    gpt_cli,
                    environment,
                    crop_path,
                    mask_path,
                    donor_path,
                    prompt,
                    item["request_size"],
                    args.quality,
                    context_image,
                    args.retries,
                )
            except RuntimeError as exc:
                raise SystemExit(
                    f"Region {name} failed; rerun with --resume after fixing: {exc}"
                ) from exc
        native, donor = normalize_donor(
            donor_path,
            item["crop_size"],
            normalized_path,
        )
        with Image.open(crop_path) as opened:
            base_crop = opened.convert("RGB")
        color_report = None
        detail_report = None
        if bool(region.get("detail_only", False)):
            donor, detail_report = transfer_detail(
                base_crop,
                donor,
                float(region.get("detail_blur_radius", args.detail_blur_radius)),
                float(region.get("detail_strength", args.detail_strength)),
                float(region.get("detail_limit", args.detail_limit)),
            )
            donor.save(work_dir / f"{index:02d}-{safe_name}-detail-transfer.png")
        elif not args.no_color_harmonization:
            donor, color_report = harmonize_donor(
                base_crop,
                donor,
                args.color_blur_radius,
                args.color_luma_limit,
                args.color_chroma_limit,
            )
            donor.save(work_dir / f"{index:02d}-{safe_name}-harmonized.png")
        delta = boundary_delta(
            base_crop,
            donor,
            edit.crop(bbox),
            int(region.get("boundary_ring", args.boundary_ring)),
        )
        if delta > args.max_boundary_delta:
            raise SystemExit(
                f"Region {name} boundary delta {delta:.3f} exceeds "
                f"{args.max_boundary_delta:.3f}; rejecting structural drift."
            )
        composite_region(
            canvas,
            donor,
            edit,
            bbox,
            int(region.get("feather", args.mask_feather)),
        )
        records.append(
            {
                "name": name,
                "pass": "interior",
                "native_dimensions": list(native),
                "normalized_dimensions": list(item["crop_size"]),
                "raw_donor": str(donor_path),
                "normalized_donor": str(normalized_path),
                "resumed": resumed,
                "boundary_delta": delta,
                "color_harmonization": color_report,
                "detail_transfer": detail_report,
                "accepted": True,
            }
        )
        if index == min(args.pilot_count, len(prepared)) and args.pilot_count:
            pilot_path = work_dir / "pilot-composite.png"
            canvas.save(pilot_path)
            (work_dir / "pilot-run.json").write_text(
                json.dumps(
                    {**plan, "records": records},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if index < len(prepared) and not args.approve_pilot:
                print(
                    f"Pilot preview written: {pilot_path}\n"
                    "Inspect it at 100%, then rerun with --resume --approve-pilot. "
                    "No final delivery was written.",
                    flush=True,
                )
                return 0
            print(
                f"Pilot approved for {index} region(s); continuing batch.",
                flush=True,
            )

    if args.edge_pass:
        for index, item in enumerate(prepared, start=1):
            region = item["region"]
            name = str(region["name"])
            safe_name = "".join(
                character if character.isalnum() else "-" for character in name
            )
            edit = item["edit"]
            bbox = item["bbox"]
            ring_width = int(region.get("edge_ring", args.edge_ring))
            ring = ImageChops.subtract(
                expand_mask(edit, ring_width), erode_mask(edit, ring_width)
            )
            ring = subtract_mask(ring, protected)
            if ring.getbbox() is None:
                continue
            crop_path = work_dir / f"{index:02d}-{safe_name}-edge-crop.png"
            mask_path = work_dir / f"{index:02d}-{safe_name}-edge-mask.png"
            donor_path = work_dir / f"{index:02d}-{safe_name}-edge-raw.png"
            normalized_path = (
                work_dir / f"{index:02d}-{safe_name}-edge-normalized.png"
            )
            canvas.crop(bbox).save(crop_path)
            mask_for_gpt(ring.crop(bbox)).save(mask_path)
            prompt = build_prompt(
                region,
                "edge",
                context_image is not None,
                args.instruction or "",
            )
            resumed = args.resume and donor_path.is_file()
            if not resumed:
                print(f"[edge {index}/{len(prepared)}] {name}", flush=True)
                try:
                    call_gpt(
                        gpt_cli,
                        environment,
                        crop_path,
                        mask_path,
                        donor_path,
                        prompt,
                        item["request_size"],
                        args.quality,
                        context_image,
                        args.retries,
                    )
                except RuntimeError as exc:
                    raise SystemExit(
                        f"Edge pass for {name} failed; rerun with --resume: {exc}"
                    ) from exc
            native, donor = normalize_donor(
                donor_path,
                item["crop_size"],
                normalized_path,
            )
            color_report = None
            detail_report = None
            if bool(region.get("detail_only", False)):
                with Image.open(crop_path) as opened:
                    base_crop = opened.convert("RGB")
                donor, detail_report = transfer_detail(
                    base_crop,
                    donor,
                    float(region.get("detail_blur_radius", args.detail_blur_radius)),
                    float(region.get("detail_strength", args.detail_strength)),
                    float(region.get("detail_limit", args.detail_limit)),
                )
                donor.save(work_dir / f"{index:02d}-{safe_name}-edge-detail-transfer.png")
            elif not args.no_color_harmonization:
                with Image.open(crop_path) as opened:
                    base_crop = opened.convert("RGB")
                donor, color_report = harmonize_donor(
                    base_crop,
                    donor,
                    args.color_blur_radius,
                    args.color_luma_limit,
                    args.color_chroma_limit,
                )
                donor.save(work_dir / f"{index:02d}-{safe_name}-edge-harmonized.png")
            composite_region(
                canvas,
                donor,
                ring,
                bbox,
                int(region.get("edge_feather", args.mask_feather)),
            )
            records.append(
                {
                    "name": name,
                    "pass": "edge",
                    "native_dimensions": list(native),
                    "normalized_dimensions": list(item["crop_size"]),
                    "raw_donor": str(donor_path),
                    "normalized_donor": str(normalized_path),
                    "resumed": resumed,
                    "color_harmonization": color_report,
                    "detail_transfer": detail_report,
                    "accepted": True,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    if source_alpha is not None:
        rgba = canvas.convert("RGBA")
        rgba.putalpha(source_alpha)
        rgba.save(output)
    else:
        canvas.save(output)
    run = {
        **plan,
        "quality": args.quality,
        "mask_expand": args.mask_expand,
        "mask_feather": args.mask_feather,
        "context_padding": args.context_padding,
        "max_boundary_delta": args.max_boundary_delta,
        "color_harmonization": not args.no_color_harmonization,
        "color_blur_radius": args.color_blur_radius,
        "color_luma_limit": args.color_luma_limit,
        "color_chroma_limit": args.color_chroma_limit,
        "detail_blur_radius": args.detail_blur_radius,
        "detail_strength": args.detail_strength,
        "detail_limit": args.detail_limit,
        "records": records,
        "model": "gpt-image-2",
        "authentication": "Codex OAuth",
        "pipeline": "semantic masks -> Only Masked crops -> deterministic composite",
    }
    (work_dir / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Completed segmented GPT redraw: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
