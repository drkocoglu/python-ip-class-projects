"""Card cropping and rank/suit segmentation.

Ports the crop-and-segment stages of the MATLAB pipeline:

1. :func:`crop_to_card`      -- from the rotated single-card image, binarize
   loosely, find the largest blob's bounding box and crop to it
   rotate 90 deg
   if the card came out landscape
   validate the aspect ratio (~1.4 for a real
   card)
   resize to a canonical ``400 x 300``.
2. :func:`extract_roi`       -- crop the top-left corner index box that holds the
   small rank glyph and suit pip.
3. :func:`segment_rank_suit` -- inside the ROI, find the two small blobs and
   assign the upper one to *rank* and the lower one to *suit*.

Canonical sizes and the area/aspect thresholds are taken directly from the
original scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import binarize, enhance, filters, morphology, regionprops, rotation


# --- canonical geometry (from the MATLAB scripts) --------------------------
CARD_SIZE = (400, 300)          # (rows, cols) after normalisation
ROI_BOX = (0, 0, 50, 130)       # (x, y, width, height) top-left corner crop
MIN_ASPECT, MAX_ASPECT = 1.3, 1.5
BLOB_AREA_MIN, BLOB_AREA_MAX = 150, 1000
BLOB_ASPECT_MIN, BLOB_ASPECT_MAX = 0.5, 1.5


@dataclass
class Segments:
    rank: np.ndarray | None
    suit: np.ndarray | None
    roi: np.ndarray
    usable: bool
    rank_bbox: tuple | None = None
    suit_bbox: tuple | None = None
    score: float = 0.0


def _resize_bilinear(image: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize to ``(rows, cols)`` -- from scratch."""
    img = np.asarray(image, dtype=np.float64)
    in_h, in_w = img.shape[:2]
    out_h, out_w = out_shape
    if in_h == out_h and in_w == out_w:
        return image.copy()

    # Map output pixel centres back to input coordinates.
    ry = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5
    rx = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    ry = np.clip(ry, 0, in_h - 1)
    rx = np.clip(rx, 0, in_w - 1)

    y0 = np.floor(ry).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x0 = np.floor(rx).astype(int)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ry - y0)[:, None]
    wx = (rx - x0)[None, :]

    Ia = img[np.ix_(y0, x0)]
    Ib = img[np.ix_(y0, x1)]
    Ic = img[np.ix_(y1, x0)]
    Id = img[np.ix_(y1, x1)]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bot * wy

    if np.issubdtype(np.asarray(image).dtype, np.integer):
        out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out


def _crop_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``(x, y, w, h)`` from an image, clipped to bounds."""
    x, y, w, h = bbox
    x0 = max(int(round(x)), 0)
    y0 = max(int(round(y)), 0)
    x1 = min(x0 + int(round(w)), image.shape[1])
    y1 = min(y0 + int(round(h)), image.shape[0])
    return image[y0:y1, x0:x1]


def crop_to_card(rotated: np.ndarray):
    """Crop a rotated single-card frame down to the normalised card image.

    Returns ``(card_400x300 | None, usable: bool)``. ``usable`` is ``False`` when
    the cropped region's aspect ratio is not card-like (blurred / bad frame).
    """
    # Binarize loosely (imbinarize(imRot, .4)) then smooth, like the original.
    bw = binarize.binarize(rotated, level=0.4)
    bw = filters.gaussian_blur(bw.astype(float), 3.0) > 0.5

    labels, n = morphology.connected_components(bw)
    if n == 0:
        return None, False
    props = regionprops.largest_region(labels, n)
    card = _crop_bbox(rotated, props.bounding_box)

    # Make the card portrait.
    if card.shape[0] < card.shape[1]:
        card = rotation.rotate(card, 90.0)
        # rotate('crop') keeps the frame size; re-crop tight to non-zero content
        card = _tight_crop(card)

    h, w = card.shape[:2]
    if w == 0 or h == 0:
        return None, False
    aspect = h / w
    if not (MIN_ASPECT < aspect < MAX_ASPECT):
        return card, False

    return _resize_bilinear(card, CARD_SIZE), True


def _tight_crop(image: np.ndarray) -> np.ndarray:
    """Crop away all-zero borders left by a 'crop' rotation."""
    nz = np.where(image > 0)
    if nz[0].size == 0:
        return image
    y0, y1 = nz[0].min(), nz[0].max()
    x0, x1 = nz[1].min(), nz[1].max()
    return image[y0:y1 + 1, x0:x1 + 1]


def extract_roi(card: np.ndarray) -> np.ndarray:
    """Crop the top-left corner index box (rank glyph + suit pip)."""
    return _crop_bbox(card, ROI_BOX)


def segment_rank_suit(roi: np.ndarray, sigma: float = 2.0) -> Segments:
    """Segment the rank (upper) and suit (lower) glyphs out of the ROI.

    Mirrors the MATLAB logic (blur, binarize, complement, keep glyph-like
    blobs
    upper blob = rank, lower = suit) with three robustness fixes:
    per-glyph area gates (a merged '10' exceeds the old 1000-px cap), border /
    speck rejection in the fallback, and same-line blob merging so multi-digit
    ranks stay whole. ``sigma`` is exposed so callers can retry with a lighter
    blur when thin glyph strokes get erased.
    """
    seg = filters.gaussian_blur(roi.astype(float), sigma)
    seg = binarize.binarize(seg)
    seg = ~seg  # imcomplement: glyphs (dark ink) become foreground

    # Card edges appear as near-full-height columns / near-full-width rows of
    # ink in the corner ROI; no glyph is that extensive. Clearing them stops
    # border lines from fusing with glyphs or winning the fallback.
    col_frac = seg.mean(axis=0)
    seg[:, col_frac > 0.8] = False
    row_frac = seg.mean(axis=1)
    seg[row_frac > 0.8, :] = False

    labels, n = morphology.connected_components(seg)
    props = regionprops.regionprops(labels, n)

    H = seg.shape[0]
    third = H / 3.0
    rank_bbox = suit_bbox = None
    for p in props:
        x, y, w, h = p.bounding_box
        aspect = w / h if h else 0
        if y > third:
            if suit_bbox is None and 150 < p.area < 1000 and 0.5 < aspect < 1.6:
                suit_bbox = p.bounding_box
        else:
            if rank_bbox is None and 150 < p.area < 1700 and 0.35 < aspect < 1.2:
                rank_bbox = p.bounding_box
        if rank_bbox is not None and suit_bbox is not None:
            break

    usable = rank_bbox is not None and suit_bbox is not None

    # Fallback so a crop is ALWAYS produced (border lines and specks excluded).
    if rank_bbox is None:
        rank_bbox = _largest_ink_bbox(seg[: int(0.5 * H)], 0)
    if suit_bbox is None:
        suit_bbox = _largest_ink_bbox(seg[int(0.42 * H):], int(0.42 * H))

    if rank_bbox is not None:
        rank_bbox = _merge_rank_digits(rank_bbox, props, roi.shape[1])

    rank = _crop_bbox(roi, rank_bbox) if rank_bbox is not None else None
    suit = _crop_bbox(roi, suit_bbox) if suit_bbox is not None else None
    segs = Segments(rank=rank, suit=suit, roi=roi, usable=usable)
    segs.rank_bbox = rank_bbox
    segs.suit_bbox = suit_bbox
    segs.score = _plausibility(rank_bbox, suit_bbox, usable, H)
    return segs


def _plausibility(rank_bbox, suit_bbox, strict: bool, H: int) -> float:
    """Geometric sanity score for a segmented corner index.

    Rewards the canonical layout: a rank glyph of sensible size sitting ABOVE a
    smaller suit pip. Used to pick between the two 180-degree orientations of a
    card and between blur scales.
    """
    score = 0.0
    if rank_bbox is not None:
        score += 2
        x, y, w, h = rank_bbox
        if 28 <= h <= 62:
            score += 1
        if 0.4 <= (w / h if h else 0) <= 1.15:
            score += 1
        if 250 <= w * h <= 2200:
            score += 1
    if suit_bbox is not None:
        score += 2
        x, y, w, h = suit_bbox
        if 14 <= h <= 36:
            score += 1
        if 0.55 <= (w / h if h else 0) <= 1.6:
            score += 1
    if rank_bbox is not None and suit_bbox is not None:
        r_cy = rank_bbox[1] + rank_bbox[3] / 2.0
        s_cy = suit_bbox[1] + suit_bbox[3] / 2.0
        score += 2 if r_cy < s_cy else -2
        if rank_bbox[3] > suit_bbox[3]:
            score += 1
    if strict:
        score += 1
    return score


def choose_segments(card: np.ndarray) -> Segments:
    """Extract the best corner segmentation, resolving the 180-degree flip.

    ``90 - orientation`` makes a card portrait but cannot tell up from down
    an
    upside-down card puts a 180-rotated index in the corner ROI, which breaks
    glyph classification. Both orientations (and a lighter blur for faint
    strokes) are segmented and the geometrically most plausible result wins.
    """
    best = None
    for flip in (False, True):
        c = card[::-1, ::-1] if flip else card
        roi = extract_roi(c)
        variants = [(roi, 2.0), (roi, 1.0)]
        # Faint ink shatters under Otsu; the project's own flowchart applies
        # adaptive histogram equalization, so a CLAHE-enhanced retry is added.
        try:
            variants.append((enhance.adapthisteq(roi, n_tiles=(2, 2)), 1.0))
        except Exception:
            pass
        for r, sigma in variants:
            segs = segment_rank_suit(r, sigma=sigma)
            if segs.rank is not None or segs.suit is not None:
                # crops must come from the un-equalised ROI for the classifier
                if r is not roi:
                    segs = Segments(
                        rank=_crop_bbox(roi, segs.rank_bbox) if segs.rank_bbox else None,
                        suit=_crop_bbox(roi, segs.suit_bbox) if segs.suit_bbox else None,
                        roi=roi, usable=segs.usable,
                        rank_bbox=segs.rank_bbox, suit_bbox=segs.suit_bbox,
                        score=segs.score)
            if best is None or segs.score > best.score:
                best = segs
            if best.score >= 10:
                return best
    return best


def _largest_ink_bbox(region_mask, row_offset):
    """Bounding box of the largest glyph-like blob in a ROI sub-region.

    Border lines (very tall/thin or full-height) and small specks are excluded
    so the fallback cannot return the card edge as a "glyph"."""
    labels, n = morphology.connected_components(region_mask)
    if n == 0:
        return None
    Hs = region_mask.shape[0]
    cands = []
    for p in regionprops.regionprops(labels, n):
        x, y, w, h = p.bounding_box
        aspect = w / h if h else 0
        if p.area >= 60 and h < 0.85 * Hs and 0.15 <= aspect <= 3.0 and w > 5:
            cands.append(p)
    if not cands:
        return None
    p = max(cands, key=lambda pr: pr.area)
    x, y, w, h = p.bounding_box
    return (x, y + row_offset, w, h)


def _merge_rank_digits(rank_bbox, props, roi_width):
    """Union the rank bbox with same-line neighbouring glyph blobs.

    A neighbour qualifies when it vertically overlaps the rank line by at least
    half its height, has glyph-like size/aspect (excludes specks and the card's
    border line), and sits within a small horizontal gap. Repeats until stable.
    """
    x0, y0, w0, h0 = rank_bbox
    bx0, by0, bx1, by1 = x0, y0, x0 + w0, y0 + h0
    for _ in range(3):
        changed = False
        h_ref = by1 - by0
        for p in props:
            qx, qy, qw, qh = p.bounding_box
            qx1, qy1 = qx + qw, qy + qh
            if qx >= bx0 and qx1 <= bx1 and qy >= by0 and qy1 <= by1:
                continue  # already inside
            ov = min(by1, qy1) - max(by0, qy)
            if ov < 0.5 * min(h_ref, qh):
                continue  # not on the same text line
            if not (0.35 * h_ref <= qh <= 1.6 * h_ref):
                continue  # too small (speck) or too tall (border line)
            aspect = qw / qh if qh else 0
            if not (0.15 <= aspect <= 2.2) or p.area < 25:
                continue
            gap = max(qx - bx1, bx0 - qx1)
            if gap > 0.8 * h_ref:
                continue  # too far to be a second digit
            nx0, ny0 = min(bx0, qx), min(by0, qy)
            nx1, ny1 = max(bx1, qx1), max(by1, qy1)
            if nx1 - nx0 > 0.9 * roi_width:
                continue  # runaway merge guard
            if (nx0, ny0, nx1, ny1) != (bx0, by0, bx1, by1):
                bx0, by0, bx1, by1 = nx0, ny0, nx1, ny1
                changed = True
        if not changed:
            break
    return (bx0, by0, bx1 - bx0, by1 - by0)
