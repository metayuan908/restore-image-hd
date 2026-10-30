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
5. Plan from a scoped actual-return profile or the conservative 1.2 MP delivery
   crop budget, including overlap. Validate the pilot's native dimensions.
6. Accept only native outputs whose width and height both cover the delivery tile.
7. Normalize down, check bounded geometry, match color, select overlap seams and
   inspect every boundary. Generators use fixed inputs; composition is serial.

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

- The default conservative budget commonly yields `1200x992` crops, including
  approximately 12.5% overlap (capped at 128 pixels per axis).
- A representative 3840-wide landscape needs roughly 12 tile calls, plus the
  global anchor for the coherent route. Actual dimensions and profiles change
  the count; use `--dry-run` rather than hardcoding it.
- `2x2` tiles of `1024x1024` cover at most `2048x2048` before overlap, not 4K.
- Overlap reduces unique coverage.

If a GPT tile returns below the delivery tile size, reduce tile dimensions and
increase tile count. Do not route the result through Real-ESRGAN or another
detail-upscaling model.

## Seam handling

Default merging chooses a continuous low-error path in a real narrow overlap;
complex overlaps explicitly report feather fallback. Explicit `core` and
`feather` modes remain available. Bounded translation is the geometry default;
optional affine requires OpenCV, small parameter bounds, and improved edge QC.
Nonlinear flow must not bend connected architecture. Color matching follows
geometry checks. Full-canvas GPT seam repair is no longer blindly dispatched:
submit a focused repair manifest for failed complete regions instead.

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
