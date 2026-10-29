# Semantic quality checks

## Industrial products

Check the complete object before judging sharpness:

- silhouette and camera perspective match the source;
- top, side, and front planes meet at physically valid corners;
- repeated frame members keep the same thickness;
- bevel radius and chamfer width remain consistent through turns;
- assembly gaps do not close, widen, float, or swap depth order;
- metal, glass, plastic, painted surfaces, and translucent materials remain distinct;
- roughness, refraction, reflection softness, and highlight width match each material;
- edge highlights continue through corners rather than ending or doubling;
- openings, handles, rails, fasteners, and recesses keep their count and alignment;
- occlusion, contact shadow, cast shadow, and floor reflection agree with the geometry;
- floor grids and perspective lines stay continuous;
- labels and logos match exactly or are rebuilt from source assets.

Reject outputs with melted corners, double edges, floating frames, impossible
reflections, pseudo-lettering, or locally sharp but globally inconsistent parts.

## People and hands

First verify the entire group:

- exact person count, identities, pose, gaze, clothing, and neighbor relationships;
- no duplicated, missing, swapped, or fused bodies;
- every arm and hand connects to the correct owner;
- wrists follow the forearm without abrupt rotation or taper;
- palm orientation and thumb side are anatomically consistent;
- visible finger count, length, joints, spacing, nails, and overlap are plausible;
- grips and hand-to-hand contacts have a clear contact point and pressure direction;
- no extra fingertips, merged fingers, phantom palms, broken sleeves, or doubled cuffs.

Create a 100% contact sheet of every face and hand. A globally pleasant image does
not pass if one hand ownership or contact relationship is wrong.

## Architecture and connected geometry

- lock vanishing points and horizon;
- preserve straight lines, module spacing, structural joints, and repeated counts;
- keep material boundaries and shadow direction continuous across tiles;
- reject locally detailed surfaces that bend the building or break alignment.

## Text, logos, UI, and diagrams

- compare exact spelling, punctuation, numbers, prices, units, and line breaks;
- preserve baseline, kerning, alignment, stroke weight, and icon geometry;
- rebuild from fonts, vectors, or source composites when exactness matters;
- never accept pseudo-text, near-match logos, invalid QR codes, or shifted UI states.

## Resolution report

Record source, global request, global native, tile request, each tile native, and
stitched delivery dimensions. A stitched 8K canvas is not a single native 8K
generation. If any tile would require pixel enlargement, fail and use smaller tiles.
