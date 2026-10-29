"""0.1.6: read the waveform axis labels (value and position) to set the scale and the zero of each waveform band.

The PB980 writes the scale of each band as grey numbers right-aligned left of the band (Paw "40" ... "0" ... "-20",
Flow "80" ... "0" ... "80" (INSP above, EXP below, without a minus sign), Volume "1000" ... "0" ... "-100").
Until 0.1.5 the scale was estimated from VTE / PPEAK and the band positions were fixed values measured on 980_sample;
a capture shifted by a few pixels (2026-09-08 videos: about 10 px at 4K) moved the zero and the scale (Flow x1.26 and
-6 L/min, Volume x1.1-1.2).

Each label (the top, zero and bottom label of a band) is compared as a whole with the labels seen before
(axis_labels.npz, made by tools/build_axis_labels.py), searched up and down from its usual row. A label is read only when
its best template is within MAX_RMS and the best template of another text is at least MIN_GAP further away. The row of
the label is the intensity centroid of the matched box (sub-pixel). A band is read when at least two labels with
different values are read and they lie on one straight line (value per pixel and zero row from a least-squares fit);
otherwise the caller keeps its estimate.

Labels are compared at BASE (1280x720) on a frame that is first scaled to the 3840x2160 reference (read_frame),
as the engine sees it. This module imports nothing from the engine except detect_sweep (tools/build_axis_labels.py uses it).
"""
import os

import cv2
import numpy as np

BASE_W, BASE_H = 1280, 720
X0, X1 = 134, 169       # label column (BASE px). Labels end at x 168; the axis line is at x 170-173
HALF = 5                # rows above and below the label row in the box
RUN_FRAC = 0.3          # rows of a label: brightest pixel at least this fraction of the label's (labels can be one blank row apart)
SEARCH = 7              # rows searched above and below the usual row
# Usual row (BASE px) of each label; band and position in the band
SLOTS = {
    "Pressure_top": 98, "Pressure_zero": 175, "Pressure_bottom": 213,
    "Flow_top": 246, "Flow_zero": 304, "Flow_bottom": 361,
    "Volume_top": 394, "Volume_zero": 498, "Volume_bottom": 508,
}
BANDS = ("Pressure", "Flow", "Volume")
MAX_RMS = 6.0           # gray levels (set from the 2026-09-08 videos and 980_sample, see tools/build_axis_labels.py check)
MIN_GAP = 4.0           # the best template of another text must be at least this much further
MAX_MASKED = 0.25       # at most this fraction of the box may be covered by a waveform
COLOR_SAT, COLOR_VAL = 60, 60   # a pixel with saturation and value at least these is part of a waveform
MAX_RESIDUAL = 1.5      # BASE px; labels of a band must lie on one line within this


def base_strip(ref_img):
    """Label column region of a 3840x2160 reference frame, scaled to BASE (INTER_AREA). Columns 0 .. X1 + 8, all rows."""
    s = ref_img.shape[0] / BASE_H
    w = int(round((X1 + 8) * s))
    return cv2.resize(ref_img[:, :w], (X1 + 8, BASE_H), interpolation=cv2.INTER_AREA)


def text_value(text, slot):
    """Label text -> value. Flow labels below zero are written without a minus sign."""
    v = float(text)
    return -abs(v) if slot == "Flow_bottom" else v


def _gray_mask(strip, y0, y1):
    c = strip[y0:y1, X0:X1]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    colour = ((hsv[..., 1] >= COLOR_SAT) & (hsv[..., 2] >= COLOR_VAL)).astype(np.uint8)
    return cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.dilate(colour, np.ones((3, 3), np.uint8)).astype(bool)


def normalized(gray, mask=None):
    """Box minus its background level (median of the pixels not covered by a waveform)."""
    keep = gray if mask is None or not mask.any() else gray[~mask]
    return gray - np.median(keep)


def centroid_row(gray, mask=None, near=None):
    """Intensity centroid row (rows from the top of the box) of the text line at row `near` (default: the middle row),
    or None. Only the rows of that line count: the run of rows around it whose brightest pixel is at least RUN_FRAC of
    the line's, so that a neighbouring label one blank row away is left out."""
    g = np.clip(normalized(gray, mask), 0, None)
    if mask is not None:
        g[mask] = 0
    peak = g.max(axis=1)
    i = len(peak) // 2 if near is None else int(near)
    lo, hi = max(0, i - 2), min(len(peak), i + 3)
    i = lo + int(np.argmax(peak[lo:hi]))     # brightest row next to the given row
    if peak[i] <= 0:
        return None
    a = b = i
    while a > 0 and peak[a - 1] >= RUN_FRAC * peak[i]:
        a -= 1
    while b < len(peak) - 1 and peak[b + 1] >= RUN_FRAC * peak[i]:
        b += 1
    rows = g[a:b + 1].sum(axis=1)
    return a + float((np.arange(len(rows)) * rows).sum() / rows.sum())


def find_label(strip, slot):
    """Box around the label of `slot`: (gray, mask, top row) centred at the text nearest to the usual row, or None.
    Neighbouring labels can be only 10 rows away, so the text is taken at the brightest row nearest to the usual row and
    its centroid is computed within CENTRE_ROWS of it."""
    y = SLOTS[slot]
    y0 = y - SEARCH - HALF
    g, m = _gray_mask(strip, y0, y + SEARCH + HALF + 1)
    prof = np.clip(normalized(g, m), 0, None)
    prof[m] = 0
    prof = prof.max(axis=1)   # brightest pixel of each row (a sum would let "-100" hide a "0" 10 rows away)
    if prof.max() <= 0:
        return None
    # rows of text: local maxima (plateaus included) of the row profile above a third of the strongest
    peaks = [i for i in range(1, len(prof) - 1) if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] >= prof.max() / 3]
    if not peaks:
        return None
    c = centroid_row(g, m, near=min(peaks, key=lambda i: abs(y0 + i - y)))
    if c is None:
        return None
    top = int(round(y0 + c)) - HALF
    g, m = _gray_mask(strip, top, top + 2 * HALF + 1)
    return g, m, top


class AxisLabelReader:
    def __init__(self, path):
        z = np.load(path)
        # templates per band: texts (one entry per template; a text can have several templates, e.g. 720p and 4K captures)
        self.sets = {}
        for band in BANDS:
            if f"texts_{band}" in z.files:
                self.sets[band] = ([str(t) for t in z[f"texts_{band}"]], z[f"templates_{band}"].astype(np.float32))

    def read_label(self, strip, slot):
        """-> (text, row in BASE px, rms) or None."""
        band = slot.split("_")[0]
        if band not in self.sets:
            return None
        texts, templates = self.sets[band]
        h = templates.shape[1]
        y = SLOTS[slot]
        y0 = y - SEARCH - HALF
        gray, mask = _gray_mask(strip, y0, y + SEARCH + HALF + 1)
        best = np.full(len(texts), np.inf)
        best_dy = np.zeros(len(texts), int)
        for dy in range(gray.shape[0] - h + 1):
            g, m = gray[dy:dy + h], mask[dy:dy + h]
            if m.mean() > MAX_MASKED:
                continue
            diff = (normalized(g, m) - templates)[:, ~m]
            rms = np.sqrt((diff ** 2).mean(axis=1))
            better = rms < best
            best[better], best_dy[better] = rms[better], dy
        order = np.argsort(best)
        i = order[0]
        if not best[i] <= MAX_RMS:
            return None
        others = [best[j] for j in order[1:] if texts[j] != texts[i]]
        if others and others[0] - best[i] < MIN_GAP:
            return None
        c = centroid_row(gray[best_dy[i]:best_dy[i] + h], mask[best_dy[i]:best_dy[i] + h])
        if c is None:
            return None
        return texts[i], y0 + best_dy[i] + c, float(best[i])

    def read(self, ref_img):
        """3840x2160 reference BGR frame -> {band: {"k": units per reference px, "zero_y": zero row, "top_y", "bottom_y",
        "top": value of the top label, "labels": {slot: value}}} for the bands that could be read."""
        if ref_img is None:
            return {}
        return self.read_strip(base_strip(ref_img), ref_img.shape[0] / BASE_H)

    def read_strip(self, strip, s):
        """BASE label strip (base_strip) and reference / BASE scale -> as read()."""
        out = {}
        for band in self.sets:
            pts = {}
            for pos in ("top", "zero", "bottom"):
                slot = f"{band}_{pos}"
                r = self.read_label(strip, slot)
                if r is not None:
                    pts[slot] = (text_value(r[0], slot), r[1])
            vals = np.array([v for v, _ in pts.values()])
            if len(set(vals)) < 2:
                continue
            ys = np.array([y for _, y in pts.values()])
            a, c = np.polyfit(ys, vals, 1)          # value = a * y + c (y down, so a < 0)
            if a >= 0 or np.abs(vals - (a * ys + c)).max() > MAX_RESIDUAL * abs(a):
                continue
            ref = lambda yb: (yb + 0.5) * s - 0.5   # BASE row -> reference row
            k = -a / s
            zero_base = -c / a
            out[band] = {"k": float(k), "zero_y": float(ref(zero_base)),
                         "top_y": float(ref(min(ys))), "bottom_y": float(ref(max(ys))),
                         "top": float(vals[np.argmin(ys)]), "labels": {sl: v for sl, (v, _) in pts.items()}}
        return out


def load_reader(engine_dir, file_name="axis_labels.npz"):
    path = os.path.join(engine_dir, file_name)
    return AxisLabelReader(path) if os.path.isfile(path) else None
