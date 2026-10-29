# GPT redraw and stitch workflows

## Route gate

Run `plan_redraw_route.py` before GPT requests. Supply semantic facts found
during visual inspection. Route high-risk connected layouts to semantic masks,
medium-risk structure to a global anchor, and low-risk texture to direct tiles.
The gate is conservative by design; escalate risk when uncertain.

## Segmented structural route

Use `gpt_segmented_redraw.py` for posters, packaging, connected foreground
subjects, exact typography layouts, and any composition where a fixed grid would
divide the main object. Read `segmented-redraw.md` for the mask manifest and
rejection rules.

Export SAM/SAM2 or manual masks, keep connected subjects together, subtract
protected text/logo masks, crop each region with context, and deterministically
composite only editable pixels. Run two pilot regions before the full batch.

## Anchored structural route

Use `gpt_coherent_redraw.py` for products, people, architecture, furniture,
vehicles, machines, or interiors when a complete anchor plus local crops is
unlikely to change the connected structure.

1. Send the complete source to GPT Image once to create a global coherence anchor.
2. Keep the original source as the authoritative local crop source.
3. Split the target canvas into smaller overlapping tiles.
4. Send each local crop as Image 1 and the complete anchor as Image 2.
5. Request the largest allowed model dimensions matching the tile ratio.
6. Accept only native outputs whose width and height both cover the delivery tile.
7. Normalize down to the tile dimensions, core-stitch, and inspect every boundary.

The first pass is not an upscaling stage. It supplies global object count,
perspective, pose, assembly, and material relationships to every local redraw.

For products, specify locked silhouette, frame thickness, bevel radius, assembly
gaps, corner joints, material roughness, refraction, reflection flow, occlusion,
and cast shadows.

For people, specify exact identities, person count, pose, clothing, hand owner,
wrist direction, palm orientation, thumb side, visible finger count, grip, and
contact relationship.

## Direct local-detail route

Use `gpt_tile_redraw.py` directly for textures, skies, foliage, terrain,
background gradients, particles, and abstract material studies. These images can
tolerate local variation.

Run `--dry-run` and review tile count before sending requests. Keep overlap,
filename, crop, and aspect ratio unchanged. Before merging, match low-frequency
luminance and chroma to the corresponding source tiles. Feather low-risk scenery,
texture, background, and abstract overlaps after color matching.

Do not bypass the structural guard for products, people, architecture, text, UI,
or diagrams. Use semantic masks when a grid boundary would cross the subject;
otherwise require a global context anchor.

## Text and brand assets

Separate semantic layers:

1. GPT-redraw photographs and textures.
2. Rebuild Chinese and Latin text with the correct font or outlines.
3. Recreate logos, rules, prices, measurements, icons, charts, QR codes, and UI
   controls deterministically.
4. Composite at final target dimensions.

Generative near-matches are not acceptable for product names, labels, prices,
legal copy, measurements, or brand marks.

## Tile geometry

- Structural 16:9 jobs default near `1536x864` tiles with about 12.5% overlap.
- Structural square jobs default near `1280x1280`.
- A 4K structural landscape usually needs about 9 GPT tile calls plus one global
  anchor call.
- An 8K structural landscape can need about 36 tile calls plus one anchor call.
- `2x2` tiles of `1024x1024` cover at most `2048x2048` before overlap, not 4K.
- Overlap reduces unique coverage.

If a GPT tile returns below the delivery tile size, reduce tile dimensions and
increase tile count. Do not route the result through Real-ESRGAN or another
detail-upscaling model.

## Seam handling

Core stitching discards conflicting overlap regions and remains the structural
default. Low-risk content uses source-anchored Y/Cb/Cr harmonization followed by
feather merge so sky, water, fog, and gradients do not show brightness or hue steps.
Full-canvas GPT seam repair is disabled because the service can return a smaller
image than requested. If explicitly enabled, a smaller repair donor is rejected,
not enlarged.

When a visible join is semantic rather than tonal—for example a broken frame,
finger, limb, edge highlight, or perspective line—rerun that subject as one
segmented mask. Split it only at a natural occlusion boundary. Feathering and
blur cannot repair incorrect structure.

## Resolution truth

Always distinguish:

- source dimensions;
- global GPT request dimensions;
- global GPT native dimensions;
- tile GPT request dimensions;
- each tile GPT native dimensions;
- stitched delivery dimensions;
- any small geometric normalization.

The delivery is a GPT-redrawn and stitched 4K/8K canvas. Do not describe it as a
single native 4K/8K generation. Do not claim interpolated pixels as recovered or
generated detail.
