---
name: restore-image-hd
description: Faithfully redraw raster images with GPT Image as mask-driven semantic regions or overlapping native-detail tiles and deliver exact HD, 2K, 4K, 8K, print, or custom-size outputs. Use for photos, connected products, people, architecture, textures, posters, packaging, scans, screenshots, or transparent assets when the user wants genuine generative reconstruction rather than ordinary upscaling. Run a conservative route gate first, use segmented Only Masked redraw for high-risk connected subjects and typography-heavy layouts, preserve alpha and sensitive text deterministically, and report GPT-native versus delivery dimensions.
---

# Restore Image HD

Create a redraw-first workflow. Do not use Real-ESRGAN, super-resolution models,
face enhancers, or interpolation as a detail-generation route. Final high-resolution
detail must come from GPT Image tile redraws and stitching.

## Execution defaults and timing

Use the existing runners; do not write task-specific opaque-output wrappers,
temporary registration pipelines, or alternative schedulers during a redraw.
Run one dependency/authentication preflight before generation. Pillow and NumPy
are required. Translation alignment uses NumPy; optional affine alignment needs
OpenCV and must be checked **before** image requests.

All three routes now share the durable segmented engine. It exports immutable
source crops before dispatch, generates with `--jobs 2`, and composes serially in
manifest priority order. A 429 reduces dispatch to one worker and receives a
bounded retry. Ambiguous network/timeout errors stop new dispatch, save completed
peers, and require `--resume`; do not blindly repeat a possibly completed POST.

Do not equate the requested image size with the native return size. Until a
representative pilot is observed, destination crops **including context** use a
conservative 1,200,000-pixel budget. This is a planning default, not a universal
model limit. Oversized semantic masks fail before requests; subdivide at natural
boundaries rather than cutting connected subjects. Runners save
`native-profile.json` from actual returns. Reuse it with `--native-profile FILE`
only for the matching backend, CLI/model, quality and local authentication
revision, within 72 hours. The first pilot still validates every native edge and
aspect ratio; an insufficient donor is never enlarged.

For opaque or transparent sources, request opaque RGB detail donors. Reapply any
original alpha deterministically. Unexpected transparent donors fail validation
instead of turning into black sky. Keep sky and other already-approved flat
areas protected when they need no new detail; retain the redraw-coverage gate.

Segmented jobs support `--stage prepare|generate|compose|all`. Preparation and
compose-only runs send no image requests. The first pilot writes
`pilot-composite.png` and pauses; inspect it at 100%, then use
`--resume --approve-pilot`. Approval applies only to the same pilot inputs and
processing settings. This is the agent's quality-review checkpoint, not a new
user-permission question. Track work from manifests and checkpoints, not chat
progress counts. Review the pilot and final QA; inspect intermediate tiles when
a check fails instead of repeatedly rebuilding the entire workflow.

`events.jsonl` records per-stage times and failures immediately. Per-job records
hash source crop, edit mask, context, prompt, native response and settings.
`--resume` reuses only matching, intact native files, and changed inputs invalidate
only affected jobs. Use `--resume --retry-region NAME` to regenerate a failed or explicitly rejected
region while retaining immutable earlier attempts and valid peers.
Old/unhashed 0.4.1 outputs are not silently trusted. No
concurrent coordinators may share a work directory. Keep raw attempts immutable.

Default geometric QC uses bounded translation. `--alignment affine` permits only
small affine corrections (2% scale/shear, 1 degree rotation, bounded shift).
Nonlinear flow is not a default restoration step: it can bend architectural
lines. Alignment precedes color matching. Large edge mismatch, missing border
coverage or excessive boundary change goes to `quality-report.json` and blocks
export; redraw only failed complete regions. `--seam-mode minimum-error` chooses
a continuous low-error path when a real narrow overlap exists. Irregular overlaps
are explicitly reported as feather fallback and still require visual review.
Neither an edge score nor a clean seam proves semantic correctness.

## Portable runtime

Resolve every script relative to this `SKILL.md`; never hardcode the source
machine's home directory. Prefer the bundled sibling
`../codex-gpt-image/scripts/codex_gpt_image.py`. Fall back to
`CODEX_GPT_IMAGE_CLI` or the user's separately installed skill only when the
bundled CLI is unavailable.

Use the target Mac's own `~/.codex/auth.json`. Never copy, package, print, or
request credentials from another machine. If authentication is missing, ask the
user to run `codex login` or the bundled device-code login.

## Inspect and classify

If the active `python3` lacks Pillow, call `load_workspace_dependencies` and use
the bundled Python executable.

1. Inspect the source at original resolution. Record dimensions, format, mode,
   alpha, visible damage, subjects, text, logos, and structural risks.
   When alpha is present, keep the source alpha mask and reapply it
   deterministically after each GPT redraw; GPT Image 2 itself returns opaque
   output.
2. Resolve the exact target:
   - 4K landscape: `3840x2160`
   - 4K portrait: `2160x3840`
   - 4K square: `4096x4096`
   - 8K landscape: `7680x4320`
   - 8K portrait: `4320x7680`
   - 8K square: `7680x7680`
   - custom: use exact width and height
3. Require the source and target aspect ratios to match within 1%. Crop or extend
   first; never stretch products, people, architecture, logos, or typography.
4. Read `references/semantic-quality.md` for products, people, hands, and text.
5. Run `scripts/plan_redraw_route.py` before any paid image request. Supply the
   semantic flags found during inspection. Treat its route as a minimum safety
   level; manually escalate risk, never downgrade it without evidence.

## Choose the redraw route

- **High-risk posters, packaging, connected products, foreground assemblies,
  people with contacts, or typography-heavy layouts:** export semantic masks and
  run `scripts/gpt_segmented_redraw.py`. Keep each connected subject inside one
  mask whenever its crop fits GPT Image limits.
- **Medium-risk products, people, architecture, or other connected structures:** run
  `scripts/gpt_coherent_redraw.py`. It first redraws the complete image as a
  global structure anchor, then redraws smaller overlapping tiles against that
  anchor and core-stitches them.
- **Textures, backgrounds, scenery, and abstract images:** run
  `scripts/gpt_tile_redraw.py` directly. These tolerate independent local detail.
- **Posters, packaging, UI, diagrams, QR codes, prices, and exact typography:**
  redraw only imagery. Rebuild or composite text, logos, rules, icons, and vector
  elements deterministically at final size.

Do not run structural content through unanchored independent tiles. Do not place
a fixed grid boundary through a connected product merely because the overlaps
look clean. A clean seam does not prove that a bevel, hand, limb, frame,
perspective line, or assembly connection is coherent.

## Mask-driven segmented redraw

Read `references/segmented-redraw.md` before this route. Accept SAM, SAM2,
Inpaint Anything, Photoshop, or manually painted grayscale masks. White means
editable; black means protected. An RGBA mask may instead use transparent pixels
as editable.

Run the route gate:

```bash
python scripts/plan_redraw_route.py poster.jpg \
  --content-mode poster \
  --preset 4k-portrait \
  --connected-subjects \
  --has-text \
  --background-only \
  --write-json work/route.json
```

Create a region manifest, review a dry plan, then run:

```bash
python scripts/gpt_segmented_redraw.py \
  poster.jpg poster-background-4k.png \
  --manifest regions.json \
  --preset 4k-portrait \
  --dry-run

python scripts/gpt_segmented_redraw.py \
  poster.jpg poster-background-4k.png \
  --manifest regions.json \
  --preset 4k-portrait \
  --edge-pass
```

The script crops each mask with context, converts it to a GPT transparent edit
mask, rejects native responses smaller than the delivery crop, checks boundary
change, matches each accepted donor's low-frequency Y/Cb/Cr response to the
authoritative source crop, and composites only the approved mask back into the
canvas. Model changes outside the mask never reach the delivery image.

Treat the first two regions as a real pilot. The initial run writes
`pilot-composite.png` and stops before the remaining requests. Inspect it at
100%, then continue with `--resume --approve-pilot`. Never infer semantic
approval from a low boundary-delta score alone.

For an enlarged delivery, require direct GPT redraw to cover at least 85% of the
canvas. Otherwise use the coherent redraw route. `--allow-source-baseline`
explicitly permits a hybrid for exceptional cases, but the result is only a 4K
canvas and must not be described as genuine 4K detail.

Keep `detail_only` disabled by default. It is an experimental source-anchored
hybrid, not a production restoration route. Use it only with
`--experimental-detail-only`, never set its strength below the quality floor,
and never use it as a silent fallback after a donor drifts.

Keep each GPT response as an immutable `*-raw.png`. Normalize into a separate
`*-normalized.png`, and record both paths plus the `resumed` flag so native
generation dimensions remain auditable across retries.

For background-only poster work, include text areas as removal regions with
instructions to continue the surrounding background. Do not ask GPT to render
replacement typography. Use protected masks when original text, logos, prices,
QR codes, or UI must remain exact.

## Coherent structural redraw

Run a dry plan first:

```bash
python scripts/gpt_coherent_redraw.py \
  input.png output-4k.png \
  --content-mode product \
  --preset 4k-landscape \
  --dry-run
```

Then run without `--dry-run`. Add a material reference only when it contributes
a specific quality. Image 1 remains authoritative:

```bash
python scripts/gpt_coherent_redraw.py \
  input.png output-4k.png \
  --content-mode product \
  --preset 4k-landscape \
  --reference material-reference.jpg \
  --instruction "Preserve IMS lettering exactly; improve metal, glass, bevels, assembly gaps, and corner continuity."
```

For people:

```bash
python scripts/gpt_coherent_redraw.py \
  people.png people-8k.png \
  --content-mode people \
  --preset 8k-landscape \
  --instruction "Preserve every identity, person count, pose, hand owner, grip, and contact point."
```

The global result is a composition anchor, not the delivered resolution. Final
detail is generated again tile by tile. The structural defaults use smaller
tiles and conservative, empirically checked request dimensions so model outputs only downsample
into delivery tiles. If either native edge is smaller than its delivery tile,
the job fails; reduce tile dimensions instead of adding an upscaler.

Use `--resume` after a failed request. For custom output:

```bash
python scripts/gpt_coherent_redraw.py \
  input.png output.png \
  --content-mode general \
  --target-width 5120 \
  --target-height 2880
```

## Direct tile redraw

Use only for low-semantic-risk content:

```bash
python scripts/gpt_tile_redraw.py \
  texture.png texture-4k.png \
  --content-mode texture \
  --preset 4k-landscape \
  --dry-run
```

Available presets include 4K and 8K landscape, portrait, and square canvases,
plus `2k-landscape`, `full-hd`, `2k-portrait`, `square-2k`,
`social-square`, `social-portrait`, `story`, and `ultrawide-4k`. Exact custom
dimensions use `--target-width` and `--target-height`.

The script requests a conservatively sized GPT Image output at each tile ratio,
normalizes it to the smaller delivery tile, discards conflicting overlap cores,
matches every tile's low-frequency Y/Cb/Cr response to the corresponding source
tile, and stitches the canvas. Low-risk scenery, texture, background, and
abstract modes feather overlaps by default; structural modes use core stitching.
It records every native tile size and color-correction report in `run.json`.
The default merge selects low-error overlap seams; `--merge-mode core` and
`--merge-mode feather` remain explicit alternatives. All routes use the same
native-size, transparency, resume and timing checks.

Structural modes require both `--allow-structural-tiling` and
`--context-image`; normally use `gpt_coherent_redraw.py`, which supplies both
safely. Full-canvas seam repair is disabled by default because GPT may return a
sub-target image. `--seam-repair` rejects such a result instead of enlarging it.

## Manual split and merge

Use `scripts/tile_image.py` when tiles are processed outside the automated flow:

```bash
python scripts/tile_image.py split \
  input.png work/tiles \
  --target-width 3840 \
  --target-height 2160 \
  --tile-width 1536 \
  --tile-height 864 \
  --overlap-x 192 \
  --overlap-y 112
```

Keep filenames, crop, aspect ratio, and boundary content unchanged. Merge
independently generated tiles with `core`; use `feather` only when processed
tiles remain geometrically identical:

```bash
python scripts/tile_image.py merge \
  work/tiles/manifest.json \
  work/tiles/processed \
  output.png \
  --mode core
```

## Validate

1. Confirm final dimensions, format, color mode, and alpha.
2. Inspect the full image and 100% crops:
   - subject count, silhouette, perspective, proportions, and repeated modules;
   - product materials, bevels, corner continuity, gaps, occlusion, reflections;
   - faces, identities, hand ownership, wrists, thumbs, fingers, grips, contacts;
   - text, logos, numbers, UI controls, diagrams, and QR codes;
   - tile boundaries, color shifts, duplicate detail, cutoffs, halos, and invented texture.
3. Compare representative crops against the source and global anchor.
4. If a structural error crosses a tile boundary, rerun the coherent route with
   segmented masks. Use smaller tiles or a stronger global instruction only when
   the semantic subject cannot fit one masked crop. Do not hide it with blur.
5. Reject the delivery when visual detail is not materially better at 100%.
   Pixel dimensions alone do not satisfy an HD, 4K, or 8K request. Regenerate a
   failed region with a revised mask or prompt; never replace it with an enlarged
   source crop.
6. Report:
   - source and delivery dimensions;
   - global GPT request and native dimensions;
   - every tile request and native dimension;
   - tile count, overlap, merge mode, and any downsampling;
   - that no detail-upscaling engine was used.

Never call the stitched delivery “single-image native 4K/8K.” Call it a
GPT-redrawn and stitched 4K/8K canvas.

## Deliver

After validation, copy only the final image to `~/Desktop/images` unless the
user requests another destination:

```bash
python scripts/deliver_image.py output.png
```

The delivery script creates the folder and avoids overwriting existing files.
User, repository and task workspace placement instructions take precedence over
that legacy default. Do not create an additional delivery directory when the
current task already specifies an outputs location.

## Offline validation

From the plugin source root, run `python -m unittest discover -s tests -v` using
the same Pillow/NumPy runtime as production. Tests use an offline fake image CLI;
they never read authentication tokens or send model requests. Keep native-size
and aspect rejection, alpha handling, input invalidation, interrupted resume,
rate-limit fallback, deterministic composition and geometric/seam cases covered.
Do not claim live backend speedup from a fake-endpoint benchmark.

## Resources

- `scripts/gpt_coherent_redraw.py`: global structure anchor followed by anchored
  native-detail tile redraw and stitching.
- `scripts/plan_redraw_route.py`: deterministic preflight risk score and route
  recommendation before any GPT Image request.
- `scripts/gpt_segmented_redraw.py`: mask-driven Only Masked crops, protected
  masks, pausing pilot gate, redraw-coverage truth gate, source-anchored color
  harmonization, boundary rejection, optional edge-ring pass, and exact
  deterministic compositing.
- `scripts/gpt_tile_redraw.py`: direct GPT Image tile redraw for low-risk content,
  native-size checks, presets, custom sizes, retry, resume, and core stitching.
- `scripts/harmonize_tiles.py`: source-anchored low-frequency luminance and chroma
  matching that removes cross-tile brightness and hue steps without GPT calls.
- `scripts/tile_image.py`: target-canvas preparation, overlapping tile export,
  core/feather merge, and manifest validation.
- `scripts/deliver_image.py`: collision-safe final delivery.
- `references/workflows.md`: routing and resolution-truth rules.
- `references/segmented-redraw.md`: mask conventions, region manifest, routing,
  poster background-only handling, and failure recovery.
- `references/semantic-quality.md`: product, material, corner, people, hand, and
  typography inspection criteria.
- `../codex-gpt-image/`: bundled Codex OAuth image CLI used on every Mac.
