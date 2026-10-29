# Mask-driven segmented redraw

## Contents

- Route selection
- Mask sources and conventions
- Region manifest
- Processing model
- Poster background-only mode
- Pilot and rejection rules
- Recovery

## Route selection

Use segmented redraw when a fixed grid would divide a connected subject or when
the image mixes photography with exact typography, logos, prices, QR codes, UI,
or diagrams. Typical cases include product posters, packaging, beverages,
furniture, machines, people touching objects, and foreground plants crossing a
main subject.

Use direct tiles only for low-risk texture, sky, fog, terrain, foliage, particles,
or abstract material. Use the global coherence route for medium-risk structure
that remains stable under local crops.

## Mask sources and conventions

Accept masks exported from SAM, SAM2, Inpaint Anything, Photoshop, Affinity,
GIMP, or manual painting. Do not bundle a large segmentation model into this
portable skill.

- Grayscale region mask: white is editable and black is protected.
- RGBA region mask: transparent is editable and opaque is protected.
- Grayscale protected mask: white is protected.
- RGBA protected mask: opaque is protected.
- Keep one complete connected subject in one mask when its padded crop is within
  the GPT Image edge and pixel limits.
- Split disconnected flowers, rocks, particles, and background areas into
  separate masks when that reduces crop size.
- Add protected masks for text, logos, prices, codes, UI, legal copy, and any
  geometry that must remain pixel-exact.

Review every mask at 100%. Reject holes through the subject, clipped silhouettes,
wrong object ownership, text leakage, and one-pixel islands before sending GPT
requests.

## Region manifest

Resolve relative paths against the manifest directory:

```json
{
  "protected_masks": [
    "masks/exact-logo.png"
  ],
  "regions": [
    {
      "name": "remove-headline",
      "mask": "masks/headline.png",
      "mode": "background",
      "pilot": true,
      "priority": 0,
      "instruction": "Remove all lettering and continue the flat mint background."
    },
    {
      "name": "cup",
      "mask": "masks/cup.png",
      "mode": "product",
      "pilot": true,
      "priority": 10,
      "expand": 8,
      "feather": 6,
      "detail_only": false,
      "instruction": "Preserve the exact cup silhouette, liquid boundary, ice count, and contact shadow."
    },
    {
      "name": "flowers",
      "mask": "masks/flowers.png",
      "mode": "foliage",
      "priority": 20,
      "instruction": "Preserve flower count, stem ownership, leaf direction, and every contact with the cup."
    }
  ]
}
```

Process lower priority values first so higher-priority foreground regions win at
overlaps. Mark representative high-risk regions with `pilot: true`; otherwise
the script uses manifest order for the first pilot set.

## Processing model

For every region:

1. Resize the mask to the exact delivery canvas with nearest-neighbor sampling.
2. Expand the mask slightly, subtract protected masks, and compute its bounding box.
3. Add 15–30% context padding without cutting connected geometry.
4. Request the largest valid native GPT Image size at the crop aspect ratio.
5. Fail if either native response edge is smaller than the delivery crop.
6. Match spatial low-frequency Y/Cb/Cr to the authoritative source crop.
7. Measure the harmonized boundary ring against the authoritative crop.
8. Reject boundary change above the configured threshold.
9. Composite only the mask into the canvas with a small feather.

Keep color harmonization enabled for aged paper, paintings, skies, gradients,
walls, and any surface where local GPT edits can create rectangular exposure or
hue steps. Disable it only when the source color is damaged and the user wants
the redraw to replace that low-frequency appearance.

The optional edge pass redraws a narrow dilated-minus-eroded ring after all
region interiors. It costs one additional GPT request per region. Use it for
hair, petals, translucent edges, product rims, contact shadows, and occlusion
boundaries; skip it for flat background fills.

Keep `detail_only` off in production. It is an experimental hybrid that preserves
source color and low-frequency structure while injecting bounded luminance
detail; it does not constitute a complete high-resolution redraw. The runner
requires `--experimental-detail-only` and rejects strengths below the configured
quality floor. Never use zero-strength or weak transfer as a fallback for a
drifting donor.

The runner never resizes the raw GPT response in place. It writes a separate
`*-normalized.png` for compositing and records both artifact paths plus whether
`--resume` reused the region. This preserves native-size evidence across
interrupted or multi-manifest runs.

## Poster background-only mode

When the user requests no fonts:

1. Detect every text, logo, price, badge, and pseudo-lettering area.
2. Create removal masks for those areas.
3. Describe the surrounding background or imagery that must continue through
   each removed area.
4. Do not request replacement letters or placeholder text.
5. Deliver the clean imagery layer only.

Typography can be rebuilt later at final resolution with actual fonts or vector
outlines. Never use a blurred source-text composite as a substitute for correct
typography.

## Pilot and rejection rules

Run two representative regions before the full batch. Include at least one
connected subject boundary and one text-removal or high-contrast edge when
present. The runner writes `pilot-composite.png` and stops. Inspect it at 100%,
then continue with `--resume --approve-pilot`.

When the output canvas enlarges the source, require direct GPT-redrawn masks to
cover at least 85% of the delivery canvas. Use the coherent redraw route when
coverage is lower. `--allow-source-baseline` is an explicit exception for a
disclosed hybrid and cannot support a genuine 4K-detail claim.

Reject a pass when:

- the native result is smaller than the delivery crop;
- the subject count, silhouette, contact, occlusion, or perspective changes;
- the boundary delta exceeds the configured threshold;
- letters, logos, prices, or watermark-like marks appear;
- content outside the intended mask reaches the delivery composite;
- the crop exceeds the GPT Image size limits.
- direct GPT redraw coverage is below the resolution-truth threshold;
- the output has 4K pixel dimensions but no material detail gain at 100%.

The script deterministically prevents the last outside-mask condition by
compositing only the mask. Visual inspection remains required for semantic
changes inside the mask.

## Recovery

- Reduce context padding before splitting a connected subject.
- Split only at natural occlusion boundaries or between disconnected components.
- Strengthen the region instruction with exact counts and relationships.
- Generate multiple candidates or revise the mask when a donor drifts; do not
  reduce detail strength to zero or silently return the enlarged source.
- Add or enlarge protected masks around typography and brand assets.
- Use `--resume` after correcting a failed mask or request.
- Disable `--edge-pass` when it invents edge texture; the accepted interior pass
  remains usable.
