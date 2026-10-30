"""
Waveform extraction for Bennett 980 engine.
Extracts Pressure, Flow, Volume waveforms from frame images
using HSV color-based pixel detection within Y-separated ROIs.

Waveform lines are cyan/teal (HSV H≈30-35) on a dark background.
Grid lines are brown/olive (HSV H≈100-105) and filtered out by hue.

Unlike NAVA, Bennett 980 does NOT have X-axis wraparound (no rewind).
Breaths that span the right edge of the waveform display are marked Fragmented.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import Settings


_config = {}

SMOOTHING = False
IMPUTATION = False

# Waveform ROI definitions (x, y, w, h) per resolution
WAVEFORM_ROIS_BY_RESOLUTION = {
    (1080, 1920): {
        "Pressure": (260, 130, 1474, 186),
        "Flow":     (260, 366, 1474, 186),
        "Volume":   (260, 586, 1474, 186),
    },
    (2160, 3840): {
        "Pressure": (520, 269, 2877, 324),
        "Flow":     (520, 733, 2877, 324),
        "Volume":   (520, 1177, 2877, 324),
    },
}

DEFAULT_WAVEFORM_ROIS = WAVEFORM_ROIS_BY_RESOLUTION[(1080, 1920)]

# HSV thresholds for waveform pixel detection
WAVEFORM_H_RANGES = [(0, 10), (20, 50), (55, 75)]
WAVEFORM_S_MIN = 80
WAVEFORM_V_MIN = 30

# Difference region detection parameters
DIFF_THRESH = 25
SMOOTH_KERNEL = 15
MIN_GAP = 30

# If the difference region extends within this many pixels of the ROI right edge,
# the breath is considered Fragmented
FRAGMENT_EDGE_MARGIN = 20

# The first breath on a new screen (the screen is redrawn from the left edge at the next inspiration after reaching the right edge) is cut from the ROI left edge
# and aligned to the start of inspiration by trim_to_breath (_snap_to_breaths). The PB980 starts drawing a new screen about 0.57 s before inspiration.
# Only when trim_to_breath is OFF, a lead-in margin in seconds is used and the breath is cut from ROI left edge + margin
# (206 px in sample.mp4 (4K, 6 s/screen = 479.5 px/s). It used to be 7.16% of the ROI width, which is 1.7 s on a 24 s screen and lost the inspiration).
NEW_SCREEN_LEAD_IN_SEC = 0.43
# If sweep_x of the start frame is within this time of the ROI left edge and not on the same screen as the previous breath, the breath is the first on a new screen
# (inspiration has already started in the start frame, so cutting from sweep_x would lose the foot of the rise)
NEW_SCREEN_MAX_SEC = 1.0

# Sweep line detection thresholds (HSV)
# The marker appears as a bright yellow-orange vertical line in all Pressure/Flow/Volume ROIs
SWEEP_H_MIN = 20
SWEEP_H_MAX = 35
SWEEP_S_MIN = 150
SWEEP_V_MIN = 100
SWEEP_MIN_COLUMN_HITS = 20   # Minimum pixels per column to count as the marker
SWEEP_MAX_ROI_DEVIATION = 15  # Maximum allowed deviation of the sweep line X between ROIs


def set_config(settings):
    global _config, SMOOTHING, IMPUTATION
    _config = settings
    # 0.1.2: both OFF by default (keep the screen trace as is). Formerly both were ON when not in the settings: columns not picked up by color were filled linearly
    # and a 10 px (4K) moving average was applied. They can be turned back on with post_ocr.smoothing / imputation
    SMOOTHING = settings.get("post_ocr", {}).get("smoothing", False)
    IMPUTATION = settings.get("post_ocr", {}).get("imputation", False)


def _get_waveform_rois(frame_shape):
    """Get waveform ROIs matching the frame resolution."""
    h, w = frame_shape[:2]
    return WAVEFORM_ROIS_BY_RESOLUTION.get((h, w), DEFAULT_WAVEFORM_ROIS)


# ============================================================
# Waveform pixel detection (HSV-based)
# ============================================================

def _extract_waveform_from_roi(hsv_img, roi, x_start=None, x_end=None):
    """
    Extract waveform Y-positions from a ROI using HSV color filtering.

    Args:
        hsv_img: HSV image (full frame)
        roi: (x, y, w, h) tuple
        x_start: optional start X within the ROI (relative to ROI left)
        x_end: optional end X within the ROI (relative to ROI left)

    Returns:
        list of float/NaN, one value per X pixel in the range
    """
    x, y, w, h = roi
    crop = hsv_img[y:y+h, x:x+w]

    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = w

    timeseries = []
    for col_x in range(x_start, x_end):
        column = crop[:, col_x, :]

        h_ch = column[:, 0]
        s_ch = column[:, 1]
        v_ch = column[:, 2]

        hue_match = np.zeros(len(h_ch), dtype=bool)
        for h_lo, h_hi in WAVEFORM_H_RANGES:
            hue_match |= (h_ch >= h_lo) & (h_ch <= h_hi)

        candidates = hue_match & (s_ch >= WAVEFORM_S_MIN) & (v_ch >= WAVEFORM_V_MIN)

        if candidates.any():
            scores = v_ch.astype(float)
            scores[~candidates] = 0
            best_y = int(np.argmax(scores))
            timeseries.append(h - best_y)
        else:
            timeseries.append(np.nan)

    return timeseries


def extract_waveform_timeseries(image_path, frame_shape=None):
    """
    Extract Pressure, Flow, Volume waveforms from a single frame image.

    Args:
        image_path: path to frame image (BGR JPEG)
        frame_shape: (h, w, ...) tuple for resolution selection.

    Returns:
        dict with keys "Pressure", "Flow", "Volume",
        each containing a numpy array of waveform Y-values.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    if frame_shape is None:
        frame_shape = img.shape

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    rois = _get_waveform_rois(frame_shape)
    waveforms = {}

    for name, roi in rois.items():
        ts = _extract_waveform_from_roi(hsv_img, roi)
        series = pd.Series(ts)
        if IMPUTATION:
            series = series.interpolate(method="linear", limit_direction="both")
        if SMOOTHING:
            series = series.rolling(window=10, center=True, min_periods=1).mean()
        waveforms[name] = series.to_numpy()

    return waveforms


# ============================================================
# Sweep line detection
# ============================================================

def detect_sweep_line(hsv_img, rois):
    """
    Detects the X position of the sweep line in a frame image.

    The sweep line appears consistently as a bright yellow-orange (H=20-35, S>=150, V>=100) vertical line
    in all Pressure/Flow/Volume ROIs.

    Args:
        hsv_img: HSV image (full frame)
        rois: waveform ROI dict {"Pressure": (x,y,w,h), ...}

    Returns:
        int or None: X position in absolute image coordinates. None if not detected.
    """
    candidates = []

    for name, (rx, ry, rw, rh) in rois.items():
        crop = hsv_img[ry:ry+rh, rx:rx+rw]

        mask = (
            (crop[:, :, 0] >= SWEEP_H_MIN) & (crop[:, :, 0] <= SWEEP_H_MAX) &
            (crop[:, :, 1] >= SWEEP_S_MIN) &
            (crop[:, :, 2] >= SWEEP_V_MIN)
        )

        col_counts = np.sum(mask, axis=0)
        min_hits = max(SWEEP_MIN_COLUMN_HITS, int(rh * 0.06))

        valid_cols = np.where(col_counts >= min_hits)[0]
        if len(valid_cols) == 0:
            return None  # fail if any ROI has no detection

        # the column with the most hits is the sweep line candidate
        best_col = valid_cols[np.argmax(col_counts[valid_cols])]
        candidates.append(best_col + rx)  # to absolute coordinates

    # check the deviation between ROIs
    if max(candidates) - min(candidates) > SWEEP_MAX_ROI_DEVIATION:
        return None

    return int(np.median(candidates))


# ============================================================
# Breath X range determination (sweep line + diff hybrid)
# ============================================================

def _determine_breath_x_range(img_start, img_end, hsv_end, rois):
    """
    Determines the X range of the active waveform of one breath.

    Strategy:
    1. Sweep line marker in the end frame -> right edge
    2. Difference between the start and end frames -> left edge
    3. Fallback: if the sweep line is not detected, the right edge is also taken from the difference

    Args:
        img_start: BGR image (start frame of the breath)
        img_end: BGR image (end frame of the breath)
        hsv_end: HSV image (end frame, already converted)
        rois: waveform ROI dict

    Returns:
        (x_left, x_right, is_fragmented, sweep_x): tuple
        - x_left: int, absolute X of the waveform left edge
        - x_right: int, absolute X of the waveform right edge
        - is_fragmented: bool, the breath reached the ROI right edge
        - sweep_x: int or None, X of the detected sweep line

        (None, None, False, None) if the range cannot be determined
    """
    sweep_x = detect_sweep_line(hsv_end, rois)
    diff_regions = get_difference_regions(img_end, img_start, rois)

    # ROI right edge (for the Fragmented check)
    roi_right_edges = [rx + rw for rx, ry, rw, rh in rois.values()]
    max_right = max(roi_right_edges)

    if sweep_x is not None:
        # Sweep line detected: right edge = sweep line
        x_right = sweep_x
        if diff_regions:
            x_left = min(r[0] for r in diff_regions)
        else:
            # If no difference is detected, use the ROI left edge
            x_left = max(rx for rx, ry, rw, rh in rois.values())
        is_fragmented = False
    elif diff_regions:
        # Fallback: difference only
        x_left = min(r[0] for r in diff_regions)
        x_right = max(r[1] for r in diff_regions)
        is_fragmented = (max_right - x_right) < FRAGMENT_EDGE_MARGIN
    else:
        return (None, None, False, None)

    if x_left >= x_right:
        return (None, None, False, None)

    return (x_left, x_right, is_fragmented, sweep_x)


# ============================================================
# Waveform extraction from contiguous X range
# ============================================================

def _extract_waveforms_in_range(hsv_img, rois, x_left, x_right):
    """
    Extracts the waveform from a contiguous X range of the end frame.

    Args:
        hsv_img: HSV image (end frame)
        rois: waveform ROI dict
        x_left: int, absolute X (start, inclusive)
        x_right: int, absolute X (end, inclusive)

    Returns:
        dict {"Pressure": numpy array, "Flow": ..., "Volume": ...}
    """
    # X range common to all ROIs (avoids length mismatches from different rx between ROIs)
    common_x_start = max(roi[0] for roi in rois.values())
    common_x_end = min(roi[0] + roi[2] for roi in rois.values())
    clipped_left = max(x_left, common_x_start)
    clipped_right = min(x_right, common_x_end - 1)

    if clipped_left >= clipped_right:
        return {name: np.array([]) for name in rois}

    waveforms = {}
    for name, roi in rois.items():
        rx, ry, rw, rh = roi
        rel_start = clipped_left - rx
        rel_end = clipped_right - rx + 1

        ts = _extract_waveform_from_roi(hsv_img, roi, rel_start, rel_end)

        series = pd.Series(ts)
        if IMPUTATION:
            series = series.interpolate(method="linear", limit_direction="both")
        if SMOOTHING:
            series = series.rolling(window=10, center=True, min_periods=1).mean()
        waveforms[name] = series.to_numpy()

    return waveforms


# ============================================================
# Difference region detection
# ============================================================

def _get_regions_for_y_range(img1, img2, y_start, y_end, x_start, x_end):
    """Detect X regions that changed between two images within a Y band."""
    gray1 = cv2.cvtColor(img1[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    column_max_diff = np.max(diff, axis=0)
    smoothed = np.convolve(column_max_diff,
                           np.ones(SMOOTH_KERNEL) / SMOOTH_KERNEL,
                           mode='same')
    x_coords = np.where(smoothed > DIFF_THRESH)[0]

    if len(x_coords) == 0:
        return []

    regions = []
    start = x_coords[0]
    for i in range(1, len(x_coords)):
        if x_coords[i] > x_coords[i - 1] + MIN_GAP:
            regions.append((start + x_start, x_coords[i - 1] + x_start))
            start = x_coords[i]
    regions.append((start + x_start, x_coords[-1] + x_start))
    return regions


def get_difference_regions(img_end, img_start, rois):
    """
    Detect X regions where waveforms changed between two frames.

    Uses all 3 waveform ROIs (Pressure, Flow, Volume) and returns
    X regions where at least 2 of the 3 show a difference.

    Args:
        img_end: BGR image of the end-of-breath frame
        img_start: BGR image of the start-of-breath frame
        rois: dict of waveform ROIs {"Pressure": (x,y,w,h), ...}

    Returns:
        list of (x_start, x_end) tuples in absolute image coordinates
    """
    # Determine common X range from the ROIs
    x_min = min(roi[0] for roi in rois.values())
    x_max = max(roi[0] + roi[2] for roi in rois.values())

    region_sets = []
    for name, (rx, ry, rw, rh) in rois.items():
        regions = _get_regions_for_y_range(
            img_start, img_end, ry, ry + rh, rx, rx + rw
        )
        region_sets.append(regions)

    # Build a bitmask and require overlap in >= 2 waveform bands
    mask = np.zeros(x_max + 1, dtype=int)
    for regions in region_sets:
        temp = np.zeros(x_max + 1, dtype=int)
        for x1, x2 in regions:
            temp[x1:x2 + 1] = 1
        mask += temp

    overlap_coords = np.where(mask >= 2)[0]
    if len(overlap_coords) == 0:
        return []

    merged = []
    start = overlap_coords[0]
    for i in range(1, len(overlap_coords)):
        if overlap_coords[i] > overlap_coords[i - 1] + 1:
            merged.append((start, overlap_coords[i - 1]))
            start = overlap_coords[i]
    merged.append((start, overlap_coords[-1]))

    return merged


def _extract_waveforms_from_regions(hsv_img, rois, regions):
    """
    Extract waveforms only from the specified X regions of the end frame.

    Uses the common X range across all ROIs so all waveforms have equal length.

    Args:
        hsv_img: HSV image of end frame
        rois: dict of waveform ROIs
        regions: list of (x_start, x_end) in absolute image coordinates

    Returns:
        dict with "Pressure", "Flow", "Volume" numpy arrays
    """
    # Compute common X range that all ROIs cover
    common_x_start = max(roi[0] for roi in rois.values())
    common_x_end = min(roi[0] + roi[2] for roi in rois.values())

    waveforms = {}
    for name, roi in rois.items():
        rx, ry, rw, rh = roi
        all_values = []
        # Process regions right-to-left so the timeseries reads chronologically
        for reg_start, reg_end in sorted(regions, key=lambda r: -r[0]):
            # Clip region to common X range, then convert to ROI-relative
            clipped_start = max(reg_start, common_x_start)
            clipped_end = min(reg_end, common_x_end - 1)
            rel_start = clipped_start - rx
            rel_end = clipped_end - rx + 1
            if rel_start >= rel_end:
                continue
            ts = _extract_waveform_from_roi(hsv_img, roi, rel_start, rel_end)
            all_values.extend(ts)

        series = pd.Series(all_values)
        if IMPUTATION:
            series = series.interpolate(method="linear", limit_direction="both")
        if SMOOTHING:
            series = series.rolling(window=10, center=True, min_periods=1).mean()
        waveforms[name] = series.to_numpy()

    return waveforms


# ============================================================
# Abnormal frame detection
# ============================================================

def DetectAbnormalFrames(user_id, df_frames, frames_root=None):
    """
    Detect abnormal frames by comparing ROI regions against a reference image.

    Args:
        user_id: str
        df_frames: pd.DataFrame with Frame column
        frames_root: str path to frame image directory (default: ./temp/{user_id}/Frames)

    Returns:
        df_frames with Abnormal, MeanDiff, StdDiff, HistCorr columns added
    """
    ref_path = Settings.REFERENCE_IMAGE_PATH
    if not os.path.exists(ref_path):
        print(f"[DetectAbnormalFrames] Reference image not found: {ref_path}, skipping")
        df_frames["Abnormal"] = "0"
        df_frames["MeanDiff"] = 0.0
        df_frames["StdDiff"] = 0.0
        df_frames["HistCorr"] = 1.0
        return df_frames

    THRESH_MEAN = 30
    THRESH_STD = 10
    THRESH_HIST_CORR = 0.90

    def compute_hist_corr(gray1, gray2):
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot read reference image: {ref_path}")

    ref_grays = []
    ref_stds = []
    for (x1, y1, x2, y2) in Settings.ROI_LIST_Abnormal_Detection:
        roi = ref_img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ref_grays.append(gray)
        ref_stds.append(np.std(gray))

    results = []
    if frames_root is None:
        frames_root = os.path.join(".", "temp", user_id, "Frames")

    for _, row in tqdm(df_frames.iterrows(), total=len(df_frames), desc="DetectAbnormalFrames"):
        frame_num = int(row["Frame"])
        filepath = os.path.join(frames_root, f"frame_{frame_num:08d}.jpg")

        img = cv2.imread(filepath)
        if img is None:
            results.append(["0", 0.0, 0.0, 0.0])
            continue

        abnormal = False
        mean_diff = 0.0
        std_diff = 0.0
        hist_corr = 1.0

        for i, (x1, y1, x2, y2) in enumerate(Settings.ROI_LIST_Abnormal_Detection):
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            mean_diff = np.mean(cv2.absdiff(ref_grays[i], gray))
            std_diff = abs(np.std(gray) - ref_stds[i])
            hist_corr = compute_hist_corr(ref_grays[i], gray)

            if mean_diff > THRESH_MEAN or std_diff > THRESH_STD or hist_corr < THRESH_HIST_CORR:
                abnormal = True
                break

        results.append(["1" if abnormal else "0", mean_diff, std_diff, hist_corr])

    df_frames[["Abnormal", "MeanDiff", "StdDiff", "HistCorr"]] = pd.DataFrame(results)
    return df_frames


# ============================================================
# Breath-level waveform CSV generation
# ============================================================

# ============================================================
# Vertical scale (pixels -> physical units)
# ============================================================
# The scale is derived per breath from the values shown at the top of the screen and the waveform pixels (same idea as NAVA).
#   Pressure: PPEAK (cmH2O) <-> peak pixel of the waveform, proportional to the height above the zero baseline (blue horizontal line)
#   Volume:   VTE (mL) <-> peak pixel of the waveform (from the zero baseline)
#   Flow:     no displayed value, so the scale is chosen so that the integral of the Flow pixels matches the change in Volume (mL)
# The estimated "value of the top scale label" is rounded to a nice value (NICE_STEPS), and outliers are removed with the values of neighboring breaths.
# Dimensions are for 4K (980_sample.mp4, measured 2026-10-11). Other resolutions are scaled by the height ratio.
#   Top scale label: Pressure 40 -> y289, Flow 120 -> y734, Volume 1000 -> y1175
#   Zero baseline:   Pressure y516.5, Flow y904.5, Volume y1486.5 (may move when the scale changes, so detected per breath)
#   Check: PPEAK 18 <-> peak 182 px -> scale 38.8 -> 40. VTE <-> peak: ratio 1.007 (+/-4%). Flow: 118.8 (117.8-119.3) -> 120
SCALE_UNITS = {"Pressure": "cmH2O", "Flow": "L/min", "Volume": "mL"}
SCALE_REF_HEIGHT = 2160
SCALE_TICK_Y = {"Pressure": 289, "Flow": 734, "Volume": 1175}
SCALE_ZERO_Y = {"Pressure": 516.5, "Flow": 904.5, "Volume": 1486.5}
ZERO_LINE_V = (28, 70)      # brightness (HSV V) of the zero baseline. The background is 10-22
NICE_STEPS = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
NICE_TOLERANCE = 0.06       # round when the estimate is within this fraction of a nice value (otherwise use the estimate as is)
SCALE_SMOOTH = 9            # remove outliers with the median of the estimates of this many neighboring breaths
SCALE_MIN_VOTES = 3         # if fewer neighboring breaths have an estimate, do not decide for this breath (use the scale of a nearby breath)
SCALE_HYSTERESIS = 0.12     # if the difference from the previous scale is below this fraction, the scale is considered unchanged
SWEEP_RATE_DEFAULT = 479.5  # sweep speed px/s (4K), used when it cannot be derived from Sweep_Start/End
FLOW_WINDOW = 12            # length (px) of the window comparing the change in Volume with the integral of Flow
# The correspondence between breath boundaries and displayed values can be off by one breath (in 980_sample the VTE in the next breath's row matches the waveform).
# The shift with the smallest spread of "displayed value / waveform peak" across breaths is chosen automatically
VALUE_SHIFTS = (0, -1)

# 0.1.6: scale and zero from the axis labels (axis_labels.py). A band whose labels are read in the end frame gets its scale
# (value per pixel) from the labels and its zero from the zero baseline (or the label "0"), and its crop covers its labels.
# Breaths where a band is not read keep the estimate above, with the label rows of the last breath read as the top label and
# zero rows, and with the values read in neighbouring breaths among the estimates (Alert "Axis").
AXIS_MARGIN = 12            # reference px kept in the crop above the top label and below the bottom label
ZERO_LINE_MAX_DIFF = 6.0    # reference px; the zero baseline gives the zero when it is this close to the label "0"
                            # (the centroid of the label "0" is 1-3 px below the baseline)
_AXIS_READER = None


def _axis_reader():
    """Reader of the axis labels (axis_labels.npz in the engine folder), or None when the templates are missing."""
    global _AXIS_READER
    if _AXIS_READER is None:
        from .axis_labels import load_reader
        _AXIS_READER = load_reader(os.path.dirname(os.path.abspath(__file__))) or False
    return _AXIS_READER or None


def _axis_rois(rois, axis_geo):
    """Crop of each band, extended to cover its axis labels (axis_geo: {band: {"top_y", "bottom_y", ...}})."""
    out = {}
    for name, (rx, ry, rw, rh) in rois.items():
        g = axis_geo.get(name)
        if g:
            y0 = min(ry, int(np.floor(g["top_y"] - AXIS_MARGIN)))
            y1 = max(ry + rh, int(np.ceil(g["bottom_y"] + AXIS_MARGIN)))
            ry, rh = y0, y1 - y0
        out[name] = (rx, ry, rw, rh)
    return out


def _zero_y(hsv_img, roi, name, ratio):
    """y of the zero baseline (blue horizontal line). Default if not found."""
    # The background is also a dark blue of the same hue (brightness 10-22), so separate by brightness. Exclude the bright borders at the top and bottom of the ROI
    x, y, w, h = roi
    m = max(4, int(round(8 * ratio)))
    band = hsv_img[y + m:y + h - m, x:x + w]
    blue = (band[:, :, 0] >= 95) & (band[:, :, 0] <= 112) & (band[:, :, 1] >= 90) \
        & (band[:, :, 2] >= ZERO_LINE_V[0]) & (band[:, :, 2] <= ZERO_LINE_V[1])
    # To avoid picking up the border of a popup in the middle of the screen (e.g. "standby pending"), only rows where the line is visible
    # near both the left and right edges are baseline candidates
    edge = max(1, int(w * 0.15))
    counts = np.minimum(blue[:, :edge].sum(axis=1), blue[:, -edge:].sum(axis=1))
    if counts.max() < 0.5 * edge:
        return SCALE_ZERO_Y[name] * ratio
    rows = np.flatnonzero(counts >= 0.5 * counts.max())
    rows = rows[np.abs(rows - rows[np.argmax(counts[rows])]) <= 3 * max(ratio, 0.5) * 2]   # only around the clearest line
    found = y + m + float(rows.mean())
    # The baseline cannot be above (or just below) the top scale label: a false detection, e.g. at a screen change
    if found <= (SCALE_TICK_Y[name] + 40) * ratio:
        return SCALE_ZERO_Y[name] * ratio
    return found


def _peak_px(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 99)) if len(v) >= 20 else None


def _nice(estimate):
    """Rounds to a nice value (if far from one, just rounds to 2 significant digits)."""
    if estimate is None or not np.isfinite(estimate) or estimate <= 0:
        return None
    exp = 10.0 ** np.floor(np.log10(estimate))
    cands = [s * exp for s in NICE_STEPS] + [10 * exp]
    best = min(cands, key=lambda c: abs(c - estimate))
    if abs(best - estimate) <= NICE_TOLERANCE * best:
        return float(best)
    return float(round(estimate, -int(np.floor(np.log10(estimate))) + 1))


def _smooth_tops(estimates, width=SCALE_SMOOTH):
    """Takes the median of the estimates (in breath order, None allowed) over width neighboring breaths, then rounds to a nice value.
    The estimate per breath varies by about +/-15%, so if the difference from the previous scale is below SCALE_HYSTERESIS, the previous scale is kept.
    Breaths that cannot be estimated (fewer than SCALE_MIN_VOTES estimates around them) are filled with the value of a nearby breath."""
    half = width // 2
    out = []
    for i in range(len(estimates)):
        window = [v for v in estimates[max(0, i - half):i + half + 1] if v is not None]
        out.append(_nice(float(np.median(window))) if len(window) >= SCALE_MIN_VOTES else None)
    prev = None
    for i, v in enumerate(out):
        if v is not None and prev is not None and abs(v - prev) < SCALE_HYSTERESIS * prev:
            out[i] = prev
        prev = out[i] if out[i] is not None else prev
    for order in (range(len(out)), range(len(out) - 1, -1, -1)):
        last = None
        for i in order:
            if out[i] is None:
                out[i] = last
            else:
                last = out[i]
    return out


def _top_estimates(records, name, values):
    """Estimate of the top scale label value from the displayed values (in breath order) and the waveform peaks (in breath order, None allowed)."""
    out = []
    for r, val in zip(records, values):
        g = r["geom"][name]
        peak = _peak_px(r["waves"][name])
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = np.nan
        if r["partial"] or peak is None or not np.isfinite(val) or val <= 0 or peak - g["zero"] < 10:
            out.append(None)
        else:
            out.append(val * (g["tick"] - g["zero"]) / (peak - g["zero"]))
    return out


def _aligned_estimates(records, name, key):
    """Estimates with the shift in VALUE_SHIFTS that gives the smallest spread (interquartile range of logs)."""
    raw = [r[key] for r in records]
    best = None
    for sh in VALUE_SHIFTS:
        vals = [raw[i - sh] if 0 <= i - sh < len(raw) else None for i in range(len(raw))]
        est = _top_estimates(records, name, vals)
        logs = np.log([e for e in est if e is not None])
        if len(logs) < 5:
            continue
        spread = float(np.subtract(*np.percentile(logs, [75, 25])))
        if best is None or spread < best[0]:
            best = (spread, est)
    return best[1] if best else [None] * len(records)


def _flow_top(r, volume_top, rate):
    """Finds the top scale label value of Flow so that the integral of Flow equals the change in Volume."""
    gv, gf = r["geom"]["Volume"], r["geom"]["Flow"]
    v = np.asarray(r["waves"]["Volume"], dtype=float)
    f = np.asarray(r["waves"]["Flow"], dtype=float) - gf["zero"]
    ml_per_px = volume_top / (gv["tick"] - gv["zero"])
    xs, ys = [], []
    for i in range(len(v) - FLOW_WINDOW):
        a, b = v[i], v[i + FLOW_WINDOW]
        seg = f[i:i + FLOW_WINDOW]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(seg).all()):
            continue
        if min(a, b) < gv["zero"] + 4 or max(a, b) > gv["h"] - 6:   # do not use Volume near 0 or stuck at the top
            continue
        xs.append(seg.sum() / rate * 1000.0 / 60.0)
        ys.append((b - a) * ml_per_px)
    if r["partial"] or len(xs) < 30:
        return None
    xs, ys = np.array(xs), np.array(ys)
    sxx = float(xs @ xs)
    return float(xs @ ys) / sxx * (gf["tick"] - gf["zero"]) if sxx > 0 else None


def _scale(name, top_value, g):
    """Top scale label value -> {min, max} corresponding to the pixel range [0, h] of the waveform CSV."""
    k = top_value / (g["tick"] - g["zero"])    # units / px
    return {"min": round(-g["zero"] * k, 2), "max": round((g["h"] - g["zero"]) * k, 2), "top": top_value,
            "unit": SCALE_UNITS[name]}


def _scale_read(name, read, g):
    """Scale of a band whose axis labels were read (read: {"k": units / px, "top": top label value})."""
    k = read["k"]
    return {"min": round(-g["zero"] * k, 2), "max": round((g["h"] - g["zero"]) * k, 2), "top": read["top"],
            "unit": SCALE_UNITS[name]}


def _with_read(records, name, estimates):
    """Estimates of the top label value, with the value read from the axis labels where the band was read (0.1.6)."""
    return [r["geom"][name]["read"]["top"] if r["geom"][name].get("read") else e for r, e in zip(records, estimates)]


def estimate_scales(records, rate):
    """{BreathNumber: {name: scale}}. records: dicts in breath order (breath, waves, geom, ppeak, vte, partial).
    Partial breaths (cut at the right edge or containing abnormal frames) are not used for the estimate and take the scale of neighboring breaths.
    0.1.6: bands whose axis labels were read (geom[name]["read"]) take the scale from the labels; their values are also
    among the estimates of the neighbouring breaths."""
    p_tops = _smooth_tops(_with_read(records, "Pressure", _aligned_estimates(records, "Pressure", "ppeak")))
    v_tops = _smooth_tops(_with_read(records, "Volume", _aligned_estimates(records, "Volume", "vte")))
    f_tops = _smooth_tops(_with_read(records, "Flow", [_flow_top(r, vt, rate) if vt else None for r, vt in zip(records, v_tops)]))
    out = {}
    for r, pt, ft, vt in zip(records, p_tops, f_tops, v_tops):
        sc = {}
        for nm, t in (("Pressure", pt), ("Flow", ft), ("Volume", vt)):
            g = r["geom"][nm]
            if g.get("read"):
                sc[nm] = _scale_read(nm, g["read"], g)
            elif t:
                sc[nm] = _scale(nm, t, g)
        out[r["breath"]] = sc
    return out


def _to_physical(values, scale, h):
    v = np.asarray(values, dtype=float)
    return np.round(scale["min"] + v / h * (scale["max"] - scale["min"]), 2)


# ============================================================
# Align breath boundaries to the start of inspiration (same idea as the trimming in NAVA v1.3)
# ============================================================
# One breath = start of inspiration to the start of the next inspiration.
# For breaths whose waveforms are contiguous in time, the boundary is moved to "TRIM_LEAD_SEC before a nearby start of inspiration",
# and rows are moved between neighboring breaths (no data is lost). At the ends of a contiguous run (no waveform in the neighboring breath),
# the part before the start of inspiration (end of the previous expiration) and a last inspiration without expiration (part of the next breath's inspiration) are trimmed.
ONSET_FLOW = 4.0        # Flow above this (L/min) is inspiratory flow
ONSET_RUN_SEC = 0.06    # inspiratory flow must last at least this long
ONSET_PEAK = 12.0       # the inspiration must reach at least this (L/min)
QUIET_SEC = 0.06        # before the inspiration, at least this long without inspiratory flow
BLIP_SEC = 0.06         # 0.1.6: a run above ONSET_FLOW this short that stays below ONSET_PEAK is noise, skipped when counting the quiet time
                        # (with the scale read from the axis labels, the pre-trigger flow of 2-4 L/min sits at ONSET_FLOW and one noisy point
                        # right before the rise hid the start of inspiration, so the flat lead-in of a new screen was not trimmed)
ONSET_FRAC = 0.2        # from the point above this fraction of the inspiratory peak, go back to the foot of the rise
EXP_FLOW = 4.0          # expiratory flow (L/min, at least this far below 0)
BACKTRACK_TOL = 0.3     # when going back to the foot, keep going while the flow keeps decreasing by at least this (L/min)
TRIM_LEAD_SEC = 0.04    # keep from this long before the start of inspiration
CONTIG_TOL_SEC = 0.15   # the breaths are contiguous if the time after the last point of the previous breath and the first point of the next breath differ by at most this (s) (allowing for timing errors)


def _inspiration_onsets(flow, rate):
    """flow: Flow (L/min, NaN allowed), rate: points/s. Row numbers of the starts of inspiration (feet of the rise)."""
    raw = np.asarray(flow, dtype=float)
    g = pd.Series(raw).interpolate(limit_area="inside").to_numpy()
    n = len(g)
    fin = np.isfinite(g)
    ins = fin & (np.nan_to_num(g) > ONSET_FLOW)
    run_min = max(2, int(round(ONSET_RUN_SEC * rate)))
    quiet_min = max(2, int(round(QUIET_SEC * rate)))
    blip_max = max(1, int(round(BLIP_SEC * rate)))
    out = []
    i = 0
    while i < n:
        if not ins[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ins[j + 1]:
            j += 1
        quiet, k = 0, i - 1
        while k >= 0 and fin[k]:
            if not ins[k]:
                quiet += 1
                k -= 1
                continue
            b = k                     # a blip: short run above ONSET_FLOW below ONSET_PEAK, with flow below ONSET_FLOW before it
            while b >= 0 and ins[b]:
                b -= 1
            if k - b > blip_max or np.nanmax(g[b + 1:k + 1]) >= ONSET_PEAK or b < 0 or not fin[b]:
                break
            k = b
        peak = np.nanmax(g[i:j + 1])
        if j - i + 1 >= run_min and peak >= ONSET_PEAK and quiet >= quiet_min:
            c = i + int(np.argmax(g[i:j + 1] >= ONSET_FRAC * peak))
            while c > 0 and fin[c - 1] and g[c - 1] < g[c] - BACKTRACK_TOL:
                c -= 1
            k = c + int(np.argmax(np.isnan(raw[c:]))) if np.isnan(raw[c:i + 1]).any() else None
            before = raw[:k][np.isfinite(raw[:k])] if k is not None else None
            if before is None or len(before) == 0 or before[-1] >= -EXP_FLOW:
                out.append(c)
        i = j + 1
    return out


def _snap_to_breaths(items):
    """items: list of dicts in time order {"wave": DataFrame (Flow column, rows = points) or None, "t0": s, "rate": points/s}.
    Writes "wave" (re-cut waveform), "t0", "rate", "trim_head", "trim_tail" (points removed from / given away from the original range; negative = received from neighbors) into each item."""
    # Split into runs of contiguous breaths
    groups, cur = [], []
    for i, it in enumerate(items):
        it.setdefault("trim_head", 0)
        it.setdefault("trim_tail", 0)
        if it["wave"] is None or len(it["wave"]) == 0 or not np.isfinite(it["t0"]):
            if cur:
                groups.append(cur)
            cur = []
            continue
        if cur:
            p = items[cur[-1]]
            if abs(p["t0"] + len(p["wave"]) / p["rate"] - it["t0"]) > CONTIG_TOL_SEC:
                groups.append(cur)
                cur = []
        cur.append(i)
    if cur:
        groups.append(cur)

    for g in groups:
        waves = [items[i]["wave"] for i in g]
        cat = pd.concat(waves, ignore_index=True)
        times = np.concatenate([items[i]["t0"] + np.arange(len(items[i]["wave"])) / items[i]["rate"] for i in g])
        bounds = np.r_[0, np.cumsum([len(w) for w in waves])]
        rate = float(np.median([items[i]["rate"] for i in g]))
        lead = max(1, int(round(TRIM_LEAD_SEC * rate)))
        flow = cat["Flow"].to_numpy(float)
        on = np.array(_inspiration_onsets(flow, rate), int)
        valid = cat[[c for c in cat.columns if c != "X" and not c.startswith("_")]].notna().any(axis=1).to_numpy()
        new = bounds.copy()
        # Head: empty rows and the part before the start of inspiration of the first breath
        lo = int(np.argmax(valid)) if valid.any() else 0
        first = on[on < (bounds[0] + bounds[1]) // 2]   # starts of inspiration in the first half of the first breath
        new[0] = max(lo, int(first[0]) - lead) if len(first) else lo
        # Boundaries between breaths: among the starts of inspiration inside the midpoints of the neighboring breaths, the one closest to the original boundary
        for j in range(1, len(g)):
            a = (bounds[j - 1] + bounds[j]) // 2
            b = (bounds[j] + bounds[j + 1]) // 2
            cand = on[(on > a) & (on < b)]
            if len(cand):
                o = int(cand[np.argmin(np.abs(cand - bounds[j]))])
                new[j] = o - lead
        # Tail: empty rows and a last inspiration without expiration (part of the next breath's inspiration)
        hi = len(cat) - int(np.argmax(valid[::-1])) if valid.any() else len(cat)
        new[-1] = hi
        last_on = on[(on > new[-2] + lead) & (on < hi)]
        if len(last_on) and not (np.nan_to_num(flow[last_on[-1]:hi]) < -EXP_FLOW).any():
            new[-1] = max(new[-2] + 1, int(last_on[-1]) - lead)
        # Keep the order (boundaries do not overtake each other)
        for j in range(1, len(new)):
            new[j] = max(new[j], new[j - 1] + 1)
        for j, i in enumerate(g):
            a, b = int(new[j]), int(new[j + 1])
            items[i]["trim_head"] = a - int(bounds[j])
            items[i]["trim_tail"] = int(bounds[j + 1]) - b
            items[i]["wave"] = cat.iloc[a:b].reset_index(drop=True)
            items[i]["t0"] = float(times[a])
            # Points received from neighboring breaths are timed at the original breath's speed, so the speed is recomputed from the times of the end points
            if b - a >= 2 and times[b - 1] > times[a]:
                items[i]["rate"] = float((b - a - 1) / (times[b - 1] - times[a]))
    return items


def _sweep_rate_from_frames(frames, fps, src_scale):
    """Sweep speed (px/s in 4K) from the median advance of Sweep_X (original video pixels) in Frames.csv between consecutive frames. None if it cannot be found."""
    if "Sweep_X" not in frames.columns:
        return None
    f = frames[["Frame", "Sweep_X"]].apply(pd.to_numeric, errors="coerce").sort_values("Frame")
    dx, dfr = f["Sweep_X"].diff().to_numpy(), f["Frame"].diff().to_numpy()
    ok = np.isfinite(dx) & (dfr == 1) & (dx > 0)   # wrap-arounds (negative) and undetected frames are not used
    if ok.sum() < 10:
        return None
    return float(np.median(dx[ok])) * fps / src_scale


def _to_source_px(df, scale):
    """Downsamples a waveform with one row per 4K pixel to one row per original video pixel (scale = original / 4K; unchanged if 1).
    Rows of the upscaled frame are interpolated, and the original video resolves only down to its own pixels."""
    if scale >= 1.0 or len(df) == 0:
        return df
    n = max(1, int(np.ceil(len(df) * scale)))
    idx = np.minimum(np.round(np.arange(n) / scale).astype(int), len(df) - 1)
    return df.iloc[idx].reset_index(drop=True)


def BreathsToCSVs(breaths, frames, frames_root, output_dir, user_id, overwrite=False):
    """
    Creates the waveform CSV of one breath based on the sweep line.

    Classified into 4 patterns by the sweep line positions in the start/end frames:
      Pattern 1: start=0, end=0 -> abnormal frame (Abnormal), skipped
      Pattern 2: start>0, end=0 -> cut at the right edge (Fragmented). 1.3.0: starts from where the previous breath ended, as in pattern 4
      Pattern 3: start=0, end>0 -> first breath on a new screen. ROI left edge (plus the lead-in margin NEW_SCREEN_LEAD_IN_SEC when trim_to_breath is OFF)
                 to the end sweep_x, analyzed in the end frame. 1.3.2: a first breath on a new screen whose start sweep_x is near the left edge
                 (pattern 4) is also cut from the same left edge
      Pattern 4: start>0, end>0 -> start sweep_x to end sweep_x, analyzed in the end frame
                 1.3.0: starts from the column after the sweep_x of the previous breath's last frame (formerly from the sweep_x of the start frame,
                 which left the waveform of one frame (about 0.1 s) between breaths in no breath)

    1.3.0: writes the time of the first point of each breath Wave_T0 (s) and Wave_Rate (px/s), and when trim_to_breath is ON (default)
    aligns the waveform boundaries to the start of inspiration (_snap_to_breaths, moving points between neighboring breaths). X is the row number from 1.

    Args:
        breaths: pd.DataFrame with BreathNumber, Frame columns
        frames: pd.DataFrame with Frame, BreathNumber, Abnormal columns
        frames_root: str path to frame image directory
        output_dir: str path to output CSV directory
        user_id: str
        overwrite: bool

    Returns:
        (effective_breaths_count, breaths)
    """
    from .detect_sweep import detect_sweep_x_img, is_settings_screen_img, read_frame, to_reference_size, source_scale, REF_W, REF_H

    os.makedirs(output_dir, exist_ok=True)

    breath_numbers = sorted(breaths["BreathNumber"].unique())
    effective = 0
    fragmented_flags = {}
    abnormal_flags = {}
    alert_map = {}
    region_x_start_map = {}
    region_x_end_map = {}
    sweep_start_map = {}
    sweep_end_map = {}
    records = []       # for the scale estimate: pixel waveforms and dimensions in breath order (converted to physical units and written after the estimate)
    sweep_rates = []
    fps = float((_config or {}).get("post_ocr", {}).get("fps", 10))
    values = breaths.set_index("BreathNumber")

    # Build the mapping source_file -> subfolder name
    def _subfolder_from_source(source_file):
        import re
        m = re.match(r"output_(\d{5})\.csv", str(source_file))
        return m.group(1) if m else None

    # ROI left edge (used in pattern 3)
    sample_path = os.path.join(frames_root, f"frame_{int(frames['Frame'].iloc[0]):08d}.jpg")
    sample_raw = cv2.imread(sample_path)
    sample_img = to_reference_size(sample_raw)
    # original video pixels / 4K pixels. Pixel values written (Sweep, Region, Wave_Rate, waveform rows) are returned to original video pixels
    src_scale = source_scale(sample_raw)
    if sample_img is not None:
        rois = _get_waveform_rois(sample_img.shape)
        roi_x_left = max(rx for rx, ry, rw, rh in rois.values())
        roi_x_right = min(rx + rw for rx, ry, rw, rh in rois.values())
    else:
        roi_x_left = 520
        roi_x_right = 520 + 2877

    # Axis labels (0.1.6): label rows of the last breath read per band, starting from the first frame
    reader = _axis_reader()

    def _read_axis(img):
        return reader.read(img) if reader is not None and img is not None and img.shape[:2] == (REF_H, REF_W) else {}

    axis_geo = dict(_read_axis(sample_img))
    axis_read = {nm: 0 for nm in SCALE_UNITS}
    axis_tops = {nm: {} for nm in SCALE_UNITS}

    # Sweep speed (4K px/s) for cutting the first breath on a new screen, from the advance of the per-frame Sweep_X of Pass 1
    early_rate = _sweep_rate_from_frames(frames, fps, src_scale) or SWEEP_RATE_DEFAULT
    trim_on = (_config or {}).get("post_ocr", {}).get("trim_to_breath", True)
    new_screen_left = roi_x_left if trim_on else roi_x_left + int(round(NEW_SCREEN_LEAD_IN_SEC * early_rate))

    # Remember a few frames already read (1.3.3). The end frame is used for sweep detection, the settings screen check and waveform extraction,
    # and as "the frame before the start frame" in the next breath, so the same image is not read again and again
    frame_cache = {}

    def _frame(path):
        if path not in frame_cache:
            if len(frame_cache) >= 4:
                frame_cache.pop(next(iter(frame_cache)))
            frame_cache[path] = read_frame(path)
        return frame_cache[path]

    # Helper to write an empty CSV when there is no waveform data
    def _write_empty_csv(path):
        pd.DataFrame(columns=["X", "Pressure", "Flow", "Volume"]).to_csv(path, index=False)

    last_end_frame = None
    for i in tqdm(range(0, len(breath_numbers)), desc="BreathsToCSVs"):
        bn = breath_numbers[i]
        is_first_breath = (i == 0)
        alert_reasons = []

        breath_frames = frames[frames["BreathNumber"] == bn].copy()
        if breath_frames.empty:
            continue

        csv_path = os.path.join(output_dir, f"{bn}.csv")
        if not overwrite and os.path.exists(csv_path):
            effective += 1
            continue

        # Flag breaths that contain Abnormal frames
        breath_frames["Abnormal"] = pd.to_numeric(breath_frames["Abnormal"], errors="coerce")
        has_abnormal = breath_frames["Abnormal"].sum() > 0
        if has_abnormal:
            abnormal_flags[bn] = True
            alert_reasons.append("Abnormal")

        start_frame_num = int(breath_frames["Frame"].min())
        end_frame_num = int(breath_frames["Frame"].max())
        prev_end_frame, last_end_frame = last_end_frame, end_frame_num

        start_path = os.path.join(frames_root, f"frame_{start_frame_num:08d}.jpg")
        end_path = os.path.join(frames_root, f"frame_{end_frame_num:08d}.jpg")

        # Detect the sweep line positions in the start/end frames
        sweep_start = detect_sweep_x_img(_frame(start_path))
        sweep_end = detect_sweep_x_img(_frame(end_path))

        def _previous_sweep():
            """sweep_x of the previous breath's last frame (= the frame before the start frame). Uses the value already detected for the previous breath."""
            prev_end = sweep_end_map.get(breath_numbers[i - 1]) if i > 0 and prev_end_frame == start_frame_num - 1 else None
            if prev_end is None:
                prev_path = os.path.join(frames_root, f"frame_{start_frame_num - 1:08d}.jpg")
                prev_end = detect_sweep_x_img(_frame(prev_path)) if os.path.isfile(prev_path) else 0  # engine_api 3: check with isfile
            return prev_end

        def _left_from_previous(x_left):
            """The column after the sweep_x of the previous breath's last frame (only on the same screen). Removes the one-frame gap between breaths.
            The first breath on a new screen (start sweep_x near the left edge and not on the same screen as the previous breath) starts from new_screen_left."""
            prev_end = _previous_sweep()
            if 0 < prev_end < x_left:
                return prev_end + 1
            if x_left - roi_x_left <= NEW_SCREEN_MAX_SEC * early_rate:
                return min(new_screen_left, x_left)
            return x_left

        sweep_start_map[bn] = sweep_start
        sweep_end_map[bn] = sweep_end

        # The first breath is treated as Fragmented (start position unknown)
        if is_first_breath:
            if sweep_end > 0:
                x_left = roi_x_left
                x_right = sweep_end
            else:
                x_left = roi_x_left
                x_right = roi_x_right
            fragmented_flags[bn] = True
            alert_reasons.append("Fragment")
        # Pattern classification
        elif sweep_start == 0 and sweep_end == 0:
            # Pattern 1: neither detected; tell the settings screen from No Sweep
            if is_settings_screen_img(_frame(end_path)):
                alert_reasons.append("Settings Change")
            else:
                alert_reasons.append("No Sweep")
            _write_empty_csv(csv_path)
            effective += 1
            alert_map[bn] = ", ".join(alert_reasons)
            continue
        elif sweep_start > 0 and sweep_end == 0:
            # Pattern 2: cut at the right edge (Fragmented) -> analyze sweep_start to the ROI right edge
            x_left = _left_from_previous(sweep_start)
            x_right = roi_x_right
            fragmented_flags[bn] = True
            alert_reasons.append("Fragment")
        elif sweep_start == 0 and sweep_end > 0:
            # Pattern 3: first breath on a new screen. ROI left edge (plus the lead-in margin when trim_to_breath is OFF) to the end sweep_x, analyzed in the end frame
            x_left = new_screen_left
            if x_left >= sweep_end:
                x_left = roi_x_left
            x_right = sweep_end
            fragmented_flags[bn] = False
        else:
            # Pattern 4: analyze start sweep_x to end sweep_x in the end frame
            x_left = sweep_start
            x_right = sweep_end
            fragmented_flags[bn] = False
            x_left = _left_from_previous(x_left)

        if x_left >= x_right:
            alert_reasons.append("Invalid Range")
            _write_empty_csv(csv_path)
            effective += 1
            alert_map[bn] = ", ".join(alert_reasons)
            continue

        # Extract the waveform from the end frame
        img_end = _frame(end_path)
        if img_end is None:
            alert_reasons.append("Frame Read Error")
            _write_empty_csv(csv_path)
            effective += 1
            alert_map[bn] = ", ".join(alert_reasons)
            continue

        axis = _read_axis(img_end)
        axis_geo.update(axis)
        rois = _axis_rois(_get_waveform_rois(img_end.shape), axis_geo)
        hsv_end = cv2.cvtColor(img_end, cv2.COLOR_BGR2HSV)
        waveforms = _extract_waveforms_in_range(hsv_end, rois, x_left, x_right)

        length = len(next(iter(waveforms.values()), []))
        if length == 0:
            alert_reasons.append("Empty Waveform")
            _write_empty_csv(csv_path)
            effective += 1
            alert_map[bn] = ", ".join(alert_reasons)
            continue

        region_x_start_map[bn] = x_left
        region_x_end_map[bn] = x_right

        df_out = pd.DataFrame({
            "X": range(1, length + 1),
            "Pressure": waveforms.get("Pressure", [np.nan] * length),
            "Flow": waveforms.get("Flow", [np.nan] * length),
            "Volume": waveforms.get("Volume", [np.nan] * length),
        })

        # Dimensions for the scale estimate (pixel value coordinates = pixels from the ROI bottom)
        ratio = img_end.shape[0] / SCALE_REF_HEIGHT
        geom = {}
        for name, (rx, ry, rw, rh) in rois.items():
            bottom = ry + rh
            zero_y, tick_y = _zero_y(hsv_end, (rx, ry, rw, rh), name, ratio), SCALE_TICK_Y[name] * ratio
            g = axis_geo.get(name)
            if g:   # label rows of this breath, or of the last breath read
                if abs(zero_y - g["zero_y"]) > ZERO_LINE_MAX_DIFF:
                    zero_y = g["zero_y"]
                tick_y = g["top_y"]
            geom[name] = {"h": rh, "zero": bottom - zero_y, "tick": bottom - tick_y}
            if name in axis:
                geom[name]["read"] = {"k": axis[name]["k"], "top": axis[name]["top"]}
                axis_read[name] += 1
                axis_tops[name][axis[name]["top"]] = axis_tops[name].get(axis[name]["top"], 0) + 1
        if reader is not None and len(axis) < len(rois):
            alert_reasons.append("Axis")
        # Position for the time of the first point (same clipping as _extract_waveforms_in_range)
        clip_left = max(x_left, max(r[0] for r in rois.values()))
        clip_right = min(x_right, min(r[0] + r[2] for r in rois.values()) - 1)
        records.append({"breath": bn, "df": df_out, "path": csv_path, "geom": geom,
                        "clip": (clip_left, clip_right), "sweep": (sweep_start, sweep_end),
                        "frames": (start_frame_num, end_frame_num),
                        "partial": bool(fragmented_flags.get(bn) or has_abnormal),
                        "waves": {nm: df_out[nm].to_numpy(float) for nm in SCALE_UNITS},
                        "ppeak": values.at[bn, "Ppeak"] if "Ppeak" in values.columns else None,
                        "vte": values.at[bn, "VTE"] if "VTE" in values.columns else None})
        if 0 < sweep_start < sweep_end and end_frame_num > start_frame_num:
            sweep_rates.append((sweep_end - sweep_start) / ((end_frame_num - start_frame_num) / fps))
        effective += 1
        if alert_reasons:
            alert_map[bn] = ", ".join(alert_reasons)

    # Estimate the scales, convert to physical units and write
    rate = float(np.median(sweep_rates)) if len(sweep_rates) >= 3 else SWEEP_RATE_DEFAULT
    scales = estimate_scales(records, rate) if records else {}
    if reader is not None and records:
        n = len(records)
        tops = "; ".join(f"{nm} " + ", ".join(f"{t:g} ({c})" for t, c in sorted(axis_tops[nm].items())) for nm in SCALE_UNITS if axis_tops[nm])
        print(f"[extract_waves] axis labels read in {n} breaths: " + ", ".join(f"{nm} {axis_read[nm]}" for nm in SCALE_UNITS)
              + (f" (top labels: {tops})" if tops else "") + "; the other bands are estimated (Alert Axis)")
    for rec in records:
        sc = scales.get(rec["breath"], {})
        for nm, scale in sc.items():
            rec["df"][nm] = _to_physical(rec["df"][nm], scale, rec["geom"][nm]["h"])

    # Time of the first point: back-calculated from the sweep_x (pen tip) of the end frame (same as the app's end-anchored timing).
    # Breaths without sweep_x in the end frame (cut off) use the sweep_x of the start frame
    wave_time = {}
    for rec in records:
        (cl, cr), (s0, s1), (f0, f1) = rec["clip"], rec["sweep"], rec["frames"]
        if s1 > 0:
            t0 = f1 / fps - (s1 - cl) / rate
        elif s0 > 0:
            t0 = f0 / fps - (s0 - cl) / rate
        else:
            t0 = f1 / fps - (len(rec["df"]) - 1) / rate
        wave_time[rec["breath"]] = t0

    # Align the waveform boundaries to the start of inspiration (breaths without a Flow scale are excluded)
    items = [{"wave": rec["df"] if "Flow" in scales.get(rec["breath"], {}) else None,
              "t0": wave_time[rec["breath"]], "rate": rate} for rec in records]
    if (_config or {}).get("post_ocr", {}).get("trim_to_breath", True):
        _snap_to_breaths(items)
    wave_info = {}
    for rec, it in zip(records, items):
        df = rec["df"] if it["wave"] is None else it["wave"]
        t0 = wave_time[rec["breath"]] if it["wave"] is None else it["t0"]
        r = rate if it["wave"] is None else it["rate"]
        df = _to_source_px(df, src_scale).copy()   # rows, speed and moved point counts in original video pixels
        df["X"] = np.arange(1, len(df) + 1)
        df.to_csv(rec["path"], index=False)
        wave_info[rec["breath"]] = (round(t0, 4), round(r * src_scale, 4),
                                    int(round(it.get("trim_head", 0) * src_scale)), int(round(it.get("trim_tail", 0) * src_scale)))
    for k, col in enumerate(("Wave_T0", "Wave_Rate", "Wave_Trim_Head", "Wave_Trim_Tail")):
        breaths[col] = breaths["BreathNumber"].map({b: v[k] for b, v in wave_info.items()})
    for nm, pfx in (("Pressure", "P"), ("Flow", "F"), ("Volume", "V")):
        for key in ("min", "max"):
            breaths[f"{pfx}_{key}"] = breaths["BreathNumber"].map(
                {b: sc[nm][key] for b, sc in scales.items() if nm in sc})

    # Add metadata columns to the Breaths DataFrame
    breaths["Fragmented"] = breaths["BreathNumber"].map(fragmented_flags).fillna(False).astype(bool)
    # Screen x is written in original video pixels (the app draws the capture sweep line etc. by the original video width)
    def _src_px(m):
        return {b: (x if src_scale == 1.0 else round(x * src_scale, 1)) for b, x in m.items()}
    breaths["Region_X_Start"] = breaths["BreathNumber"].map(_src_px(region_x_start_map)).fillna("")
    breaths["Region_X_End"] = breaths["BreathNumber"].map(_src_px(region_x_end_map)).fillna("")
    breaths["Sweep_Start"] = breaths["BreathNumber"].map(_src_px(sweep_start_map)).fillna("")
    breaths["Sweep_End"] = breaths["BreathNumber"].map(_src_px(sweep_end_map)).fillna("")

    # Set the Alert column
    breaths["Alert"] = breaths["BreathNumber"].map(alert_map).fillna("")

    # Build Start_File_Name / End_File_Name
    # The label screen expects the "subfolder/frame_XXXXXXXX.jpg" format,
    # so build it from iFrame (local frame number) and source_file (output CSV name)
    start_file_names = {}
    end_file_names = {}
    for bn in breath_numbers:
        breath_frames = frames[frames["BreathNumber"] == bn]
        if breath_frames.empty:
            continue
        # Start frame row
        start_row = breath_frames.loc[breath_frames["Frame"].idxmin()]
        subfolder = _subfolder_from_source(start_row.get("source_file", ""))
        iframe = int(start_row.get("iFrame", start_row["Frame"]))
        if subfolder:
            start_file_names[bn] = f"{subfolder}/frame_{iframe:08d}.jpg"

        # End frame row
        end_row = breath_frames.loc[breath_frames["Frame"].idxmax()]
        subfolder = _subfolder_from_source(end_row.get("source_file", ""))
        iframe = int(end_row.get("iFrame", end_row["Frame"]))
        if subfolder:
            end_file_names[bn] = f"{subfolder}/frame_{iframe:08d}.jpg"

    breaths["Start_File_Name"] = breaths["BreathNumber"].map(start_file_names).fillna("")
    breaths["End_File_Name"] = breaths["BreathNumber"].map(end_file_names).fillna("")

    return effective, breaths
