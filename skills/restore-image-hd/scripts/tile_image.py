#!/usr/bin/env python3
"""Split an image into overlapping tiles and merge processed tiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Pillow and NumPy are required: python3 -m pip install pillow numpy"
    ) from exc


MANIFEST_NAME = "manifest.json"


def positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    points = list(range(0, length - tile + 1, stride))
    last = length - tile
    if points[-1] != last:
        points.append(last)
    return points


def load_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read manifest {path}: {exc}") from exc
    required = {"canvas_width", "canvas_height", "tiles"}
    missing = sorted(required - data.keys())
    if missing:
        raise SystemExit(f"Manifest is missing fields: {', '.join(missing)}")
    return data


def split_command(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input image not found: {source}")
    tile_width = args.tile_width or args.tile_size or 1024
    tile_height = args.tile_height or args.tile_size or 1024
    overlap_x = (
        args.overlap_x
        if args.overlap_x is not None
        else args.overlap if args.overlap is not None else 160
    )
    overlap_y = (
        args.overlap_y
        if args.overlap_y is not None
        else args.overlap if args.overlap is not None else 160
    )
    if overlap_x < 0 or overlap_x >= tile_width:
        raise SystemExit("Horizontal overlap must be >= 0 and smaller than tile width.")
    if overlap_y < 0 or overlap_y >= tile_height:
        raise SystemExit("Vertical overlap must be >= 0 and smaller than tile height.")

    with Image.open(source) as opened:
        image = opened.convert("RGBA")
        source_width, source_height = image.size

    width = args.target_width or source_width
    height = args.target_height or source_height
    if width <= 0 or height <= 0:
        raise SystemExit("Target dimensions must be positive.")
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    output_dir.mkdir(parents=True, exist_ok=True)
    x_positions = positions(width, tile_width, overlap_x)
    y_positions = positions(height, tile_height, overlap_y)
    tiles: list[dict] = []

    index = 1
    for row, y in enumerate(y_positions):
        for column, x in enumerate(x_positions):
            right = min(x + tile_width, width)
            bottom = min(y + tile_height, height)
            filename = f"tile-r{row:02d}-c{column:02d}-{index:03d}.png"
            tile_image = image.crop((x, y, right, bottom))
            tile_image.save(output_dir / filename)
            tiles.append(
                {
                    "id": index,
                    "row": row,
                    "column": column,
                    "x": x,
                    "y": y,
                    "width": right - x,
                    "height": bottom - y,
                    "file": filename,
                }
            )
            index += 1

    manifest = {
        "version": 1,
        "source": str(source),
        "source_width": source_width,
        "source_height": source_height,
        "canvas_width": width,
        "canvas_height": height,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "overlap_x": overlap_x,
        "overlap_y": overlap_y,
        "tile_size": tile_width if tile_width == tile_height else None,
        "overlap": overlap_x if overlap_x == overlap_y else None,
        "resized_with": "Lanczos" if (width, height) != (source_width, source_height) else None,
        "tiles": tiles,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Canvas: {width}x{height}")
    print(f"Grid: {len(x_positions)} columns x {len(y_positions)} rows")
    print(
        f"Tiles: {len(tiles)}; size={tile_width}x{tile_height}; "
        f"overlap={overlap_x}x{overlap_y}"
    )
    print(f"Manifest: {manifest_path}")
    return 0


def axis_weight(length: int, overlap: int, fade_start: bool, fade_end: bool) -> np.ndarray:
    weight = np.ones(length, dtype=np.float32)
    fade = min(overlap, length)
    if fade and fade_start:
        weight[:fade] *= np.linspace(1e-3, 1.0, fade, dtype=np.float32)
    if fade and fade_end:
        weight[-fade:] *= np.linspace(1.0, 1e-3, fade, dtype=np.float32)
    return weight


def axis_core_bounds(tiles: list[dict], axis: str, canvas_length: int) -> dict[int, tuple[int, int]]:
    position_key = axis
    size_key = "width" if axis == "x" else "height"
    positions = sorted({int(tile[position_key]) for tile in tiles})
    extents = {
        position: max(
            int(tile[position_key]) + int(tile[size_key])
            for tile in tiles
            if int(tile[position_key]) == position
        )
        for position in positions
    }
    boundaries: list[int] = []
    for left_position, right_position in zip(positions, positions[1:]):
        left_extent = extents[left_position]
        if left_extent < right_position:
            raise SystemExit(
                f"Tiles have a gap on the {axis}-axis: {left_extent}..{right_position}."
            )
        boundaries.append((left_extent + right_position) // 2)

    bounds: dict[int, tuple[int, int]] = {}
    for index, position in enumerate(positions):
        start = 0 if index == 0 else boundaries[index - 1]
        end = canvas_length if index == len(positions) - 1 else boundaries[index]
        bounds[position] = (start, end)
    return bounds


def core_merge(manifest: dict, tiles_dir: Path) -> Image.Image:
    width = int(manifest["canvas_width"])
    height = int(manifest["canvas_height"])
    tiles = manifest["tiles"]
    x_bounds = axis_core_bounds(tiles, "x", width)
    y_bounds = axis_core_bounds(tiles, "y", height)
    canvas = Image.new("RGBA", (width, height))

    for tile_info in tiles:
        tile_path = tiles_dir / tile_info["file"]
        if not tile_path.is_file():
            raise SystemExit(f"Processed tile is missing: {tile_path}")
        x = int(tile_info["x"])
        y = int(tile_info["y"])
        expected = (int(tile_info["width"]), int(tile_info["height"]))
        with Image.open(tile_path) as opened:
            tile = opened.convert("RGBA")
        if tile.size != expected:
            raise SystemExit(
                f"Tile {tile_path.name} is {tile.size}, expected {expected}. "
                "Do not change tile dimensions during processing."
            )

        core_left, core_right = x_bounds[x]
        core_top, core_bottom = y_bounds[y]
        crop_box = (
            core_left - x,
            core_top - y,
            core_right - x,
            core_bottom - y,
        )
        if (
            crop_box[0] < 0
            or crop_box[1] < 0
            or crop_box[2] > tile.width
            or crop_box[3] > tile.height
        ):
            raise SystemExit(f"Core crop exceeds tile bounds for {tile_path.name}.")
        canvas.paste(tile.crop(crop_box), (core_left, core_top))
    return canvas


def merge_command(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    tiles_dir = args.tiles_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output}. Pass --force to replace it.")

    manifest = load_manifest(manifest_path)
    width = int(manifest["canvas_width"])
    height = int(manifest["canvas_height"])
    if args.mode == "core":
        image = core_merge(manifest, tiles_dir)
        if image.getextrema()[3] == (255, 255):
            image = image.convert("RGB")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        print(f"Merged: {len(manifest['tiles'])} tiles; mode=core")
        print(f"Output: {width}x{height} {image.mode}")
        print(f"Saved: {output}")
        return 0

    overlap_x = int(manifest.get("overlap_x") or manifest.get("overlap") or 0)
    overlap_y = int(manifest.get("overlap_y") or manifest.get("overlap") or 0)
    accumulator = np.zeros((height, width, 4), dtype=np.float32)
    weight_sum = np.zeros((height, width, 1), dtype=np.float32)

    for tile_info in manifest["tiles"]:
        tile_path = tiles_dir / tile_info["file"]
        if not tile_path.is_file():
            raise SystemExit(f"Processed tile is missing: {tile_path}")
        x = int(tile_info["x"])
        y = int(tile_info["y"])
        expected = (int(tile_info["width"]), int(tile_info["height"]))
        with Image.open(tile_path) as opened:
            tile = opened.convert("RGBA")
        if tile.size != expected:
            raise SystemExit(
                f"Tile {tile_path.name} is {tile.size}, expected {expected}. "
                "Do not change tile dimensions during processing."
            )

        tile_array = np.asarray(tile, dtype=np.float32)
        tile_width, tile_height = tile.size
        horizontal = axis_weight(
            tile_width, overlap_x, fade_start=x > 0, fade_end=x + tile_width < width
        )
        vertical = axis_weight(
            tile_height, overlap_y, fade_start=y > 0, fade_end=y + tile_height < height
        )
        weight = (vertical[:, None] * horizontal[None, :])[:, :, None]
        accumulator[y : y + tile_height, x : x + tile_width] += tile_array * weight
        weight_sum[y : y + tile_height, x : x + tile_width] += weight

    if np.any(weight_sum <= 0):
        raise SystemExit("Tile manifest does not fully cover the target canvas.")
    merged = np.rint(np.clip(accumulator / weight_sum, 0, 255)).astype(np.uint8)
    image = Image.fromarray(merged, mode="RGBA")
    if np.all(merged[:, :, 3] == 255):
        image = image.convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Merged: {len(manifest['tiles'])} tiles; mode=feather")
    print(f"Output: {width}x{height} {image.mode}")
    print(f"Saved: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and merge overlapping image tiles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split", help="Resize a canvas and export tiles.")
    split.add_argument("input", type=Path)
    split.add_argument("output_dir", type=Path)
    split.add_argument("--target-width", type=int)
    split.add_argument("--target-height", type=int)
    split.add_argument("--tile-size", type=int)
    split.add_argument("--tile-width", type=int)
    split.add_argument("--tile-height", type=int)
    split.add_argument("--overlap", type=int)
    split.add_argument("--overlap-x", type=int)
    split.add_argument("--overlap-y", type=int)
    split.set_defaults(function=split_command)

    merge = subparsers.add_parser("merge", help="Merge processed tiles.")
    merge.add_argument("manifest", type=Path)
    merge.add_argument("tiles_dir", type=Path)
    merge.add_argument("output", type=Path)
    merge.add_argument(
        "--mode",
        choices=("feather", "core"),
        default="feather",
        help="Feather overlaps or keep only the central core of each tile.",
    )
    merge.add_argument("--force", action="store_true")
    merge.set_defaults(function=merge_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    sys.exit(main())
