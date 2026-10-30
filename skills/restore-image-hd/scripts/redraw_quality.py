"""Geometry-first normalization and seam selection; never creates HD detail."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter


def inspect_native(path, expected):
    with Image.open(path) as opened:
        opened.load()
        size = opened.size
        if size[0] < expected[0] or size[1] < expected[1]:
            raise RuntimeError(f"Native {size[0]}x{size[1]} is below delivery crop {expected[0]}x{expected[1]}; replan smaller regions, never enlarge")
        if abs(size[0] / size[1] / (expected[0] / expected[1]) - 1) > 0.01:
            raise RuntimeError("Native aspect ratio drift exceeds 1%; refusing to stretch geometry")
        if "A" in opened.getbands() and opened.getchannel("A").getextrema() != (255, 255):
            raise RuntimeError("Unexpected transparent native result; refusing black RGB conversion. Request --background opaque; source alpha is reapplied separately")
        return {"native_dimensions": list(size), "native_mode": opened.mode}


def normalize_native(path, expected, output):
    info = inspect_native(path, expected)
    with Image.open(path) as opened:
        donor = opened.convert("RGB")
    if donor.size != tuple(expected):
        donor = donor.resize(tuple(expected), Image.Resampling.LANCZOS)
    donor.save(output)
    return tuple(info["native_dimensions"]), donor


def _gray(image, size=None):
    image = image.convert("L")
    if size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)


def phase_translation(base, donor):
    scale = min(1.0, 768 / max(base.size))
    size = tuple(max(16, round(v * scale)) for v in base.size)
    a, b = _gray(base, size), _gray(donor, size)
    if a.std() < 2 or b.std() < 2:
        return (0.0, 0.0), 0.0
    window = np.outer(np.hanning(size[1]), np.hanning(size[0]))
    fa, fb = np.fft.fft2((a-a.mean()) * window), np.fft.fft2((b-b.mean()) * window)
    cross = fa * np.conj(fb)
    corr = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-9)).real
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    if x > size[0] // 2:
        x -= size[0]
    if y > size[1] // 2:
        y -= size[1]
    confidence = float(np.max(corr) / max(np.mean(np.abs(corr)), 1e-9))
    return (x / scale, y / scale), confidence


def edge_distance(base, donor):
    """Reference strong-edge -> donor edge p95, with bounded memory, in pixels.

    Added fine texture is not treated as a geometric defect. This is a triage
    metric and does not replace semantic visual review.
    """
    scale = min(1.0, 512 / max(base.size))
    size = tuple(max(16, round(v * scale)) for v in base.size)
    def edges(image):
        y, x = np.gradient(_gray(image, size))
        magnitude = np.hypot(x, y)
        threshold = max(4.0, float(np.percentile(magnitude, 90)))
        return magnitude > threshold
    a, b = edges(base), edges(donor)
    # Ignore image borders affected by cropping or interpolation.
    a[:4] = a[-4:] = False
    a[:, :4] = a[:, -4:] = False
    if a.sum() < 16:
        return {"p95_pixels": 0.0, "informative": False}
    distance = np.where(b, 0.0, 32.0).astype(np.float32)
    for _ in range(24):
        old = distance
        padded = np.pad(old, 1, constant_values=32)
        distance = np.minimum.reduce([old, padded[:-2,1:-1]+1, padded[2:,1:-1]+1,
                                      padded[1:-1,:-2]+1, padded[1:-1,2:]+1])
    return {"p95_pixels": float(np.percentile(distance[a], 95) / scale), "informative": True}


def align_donor(base, donor, mode="translation", max_shift=24.0, max_edge_distance=5.0):
    before = edge_distance(base, donor)
    report = {"mode": mode, "before": before, "applied": False}
    if before["informative"] and before["p95_pixels"] <= max_edge_distance:
        return donor, {**report, "after": before, "reason": "already within geometry tolerance"}
    if mode == "none":
        result = donor
    else:
        (dx, dy), confidence = phase_translation(base, donor)
        report.update(translation=[dx, dy], confidence=confidence)
        if max(abs(dx), abs(dy)) > max_shift:
            raise RuntimeError(f"Required alignment shift {dx:.1f},{dy:.1f} exceeds {max_shift}; regenerate the complete region")
        matrix = np.array([[1.,0.,-dx],[0.,1.,-dy]], dtype=np.float32)
        if mode == "affine":
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("Affine alignment needs OpenCV; install it during preflight or use translation") from error
            try:
                _, matrix = cv2.findTransformECC(_gray(base)/255, _gray(donor)/255,
                                                 matrix, cv2.MOTION_AFFINE,
                                                 (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 80, 1e-5),
                                                 None, 5)
            except cv2.error as error:
                raise RuntimeError("Affine alignment did not converge; regenerate region") from error
            singular = np.linalg.svd(matrix[:,:2], compute_uv=False)
            angle = abs(float(np.degrees(np.arctan2(matrix[1,0], matrix[0,0]))))
            if np.any(np.abs(singular-1) > .02) or angle > 1 or np.max(np.abs(matrix[:,2])) > max_shift:
                raise RuntimeError("Affine exceeds 2% scale/shear, 1 degree rotation or translation limit")
        if confidence < 6 and mode == "translation":
            # Never invent an alignment from an uninformative phase peak.
            matrix = np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32)
        # One bounded affine mapping; no nonlinear flow that bends architecture.
        result = donor.transform(base.size, Image.Transform.AFFINE, tuple(matrix.ravel()),
                                 resample=Image.Resampling.BICUBIC)
        # Reject shifts requiring pixels beyond native crop coverage. Output is
        # only used under an eroded valid selector by the caller.
        report.update(matrix=matrix.tolist(), applied=not np.allclose(matrix, [[1,0,0],[0,1,0]]))
    after = edge_distance(base, result)
    if report["applied"] and after["p95_pixels"] > before["p95_pixels"]:
        result, after = donor, before
        report.update(applied=False, rejected_alignment="edge distance worsened")
    report["after"] = after
    if after["informative"] and after["p95_pixels"] > max_edge_distance:
        raise RuntimeError(f"Geometric edge mismatch {after['p95_pixels']:.1f}px exceeds {max_edge_distance}; regenerate region instead of hiding it with blur")
    return result, report


def alignment_valid_mask(size, report):
    """Hard validity mask after an affine map; prevents black crop borders."""
    mask = Image.new("L", size, 255)
    if report.get("applied"):
        matrix = np.asarray(report["matrix"])
        mask = mask.transform(size, Image.Transform.AFFINE, tuple(matrix.ravel()),
                              resample=Image.Resampling.NEAREST, fillcolor=0)
    return mask


def minimum_error_selector(old, new, selector, coverage):
    """Choose a continuous seam inside a real, narrow rectangular overlap.

    Irregular/multiple overlaps explicitly fall back to the original selector;
    they remain subject to geometry/boundary QC, not a false seam-success claim.
    """
    edit = np.asarray(selector) > 127
    prior = np.asarray(coverage) > 127
    overlap = edit & prior
    if not overlap.any() or not (edit & ~prior).any():
        return selector, {"used": False, "reason": "no shared overlap or complete replacement"}
    ys, xs = np.where(overlap)
    x0,x1,y0,y1 = xs.min(),xs.max()+1,ys.min(),ys.max()+1
    horizontal = (y1-y0) < (x1-x0)
    # Work in (long-axis, overlap-width) coordinates.
    band = overlap[y0:y1,x0:x1]
    a=np.asarray(old.convert("RGB"),dtype=np.float32)[y0:y1,x0:x1]
    b=np.asarray(new.convert("RGB"),dtype=np.float32)[y0:y1,x0:x1]
    cost=np.mean(np.abs(a-b),axis=2)
    if horizontal:
        band,cost=band.T,cost.T
    if band.shape[1] > 256 or not np.all(band.any(axis=1)):
        return selector, {"used": False, "reason": "irregular overlap; inspect explicit boundary QA"}
    cost=np.where(band,cost,np.inf)
    parent=np.zeros(cost.shape,np.int8)
    acc=cost[0].copy()
    for row in range(1,len(cost)):
        choices=np.stack([np.r_[np.inf,acc[:-1]],acc,np.r_[acc[1:],np.inf]])
        best=np.argmin(choices,axis=0)
        parent[row]=best-1
        acc=cost[row]+np.min(choices,axis=0)
    if not np.isfinite(acc).any():
        return selector,{"used":False,"reason":"no continuous overlap path"}
    route=np.empty(len(cost),int);route[-1]=int(np.argmin(acc))
    for row in range(len(cost)-1,0,-1):
        route[row-1]=route[row]+parent[row,route[row]]
    fresh_y,fresh_x=np.where(edit & ~prior)
    new_high=(fresh_y.mean()>(y0+y1)/2) if horizontal else (fresh_x.mean()>(x0+x1)/2)
    ownership=np.arange(cost.shape[1])[None,:]>=route[:,None]
    if not new_high:
        ownership=~ownership
    if horizontal:
        ownership=ownership.T
    result=np.array(selector)
    view=result[y0:y1,x0:x1]
    view[overlap[y0:y1,x0:x1]]=ownership[overlap[y0:y1,x0:x1]]*255
    return Image.fromarray(result),{"used":True,"axis":"horizontal" if horizontal else "vertical",
                                   "mean_path_error":float(np.mean(cost[np.arange(len(route)),route]))}
