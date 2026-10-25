"""
Bennett 980 sweep line detection utility.

Given a JPEG frame image, returns the X position of the sweep line.
Returns 0 if it cannot be detected.

How the sweep line is detected:
  In all three ROIs (Pressure/Flow/Volume), excluding grid lines (H=90-130),
  a band of at least MIN_SWEEP_WIDTH consecutive columns whose V_max is at or below the threshold = the sweep line.
  If the band is wider than 10 px, the range of 10 px from its left edge is returned.

Usage:
  python detect_sweep.py <frame_path>
  python detect_sweep.py /opt/ventocr/app/temp/admin/Frames/00001/frame_00000461.jpg
"""

import sys
import cv2
import numpy as np


# ROI definitions for 3840x2160 (X start = 520 and width = 2877 for all)
ROIS = [
    ("Pressure", 269, 324),  # (y, h)
    ("Flow",     733, 324),
    ("Volume",  1177, 324),
]
ROI_X = 520
ROI_W = 2877

# Minimum width (px) of the sweep band. 11-14 px in 4K captures, 8-11 px in 1280x720 captures upscaled to 4K
MIN_SWEEP_WIDTH = 7

# Upper limit of V_max inside the sweep band (after excluding grid lines)
VMAX_THRESH = 45

# H range of grid lines (excluded from the V_max calculation)
GRID_H_MIN = 90
GRID_H_MAX = 130

# Minimum number of waveform columns per ROI (abnormal frame check)
MIN_WAVE_COLS_PER_ROI = 100

# HSV conditions for waveform pixels
WAVE_H_MIN = 55
WAVE_H_MAX = 75
WAVE_S_MIN = 80
WAVE_V_MIN = 30

# All engine coordinates and thresholds are defined on 3840x2160 frames
REF_W, REF_H = 3840, 2160


def to_reference_size(img):
    """Upscales 16:9 frames other than 3840x2160 (1280x720, 1920x1080, etc.) to 3840x2160.
    Videos captured directly from the PB980 screen have the same layout at any resolution, so the 4K coordinates and thresholds apply after upscaling.
    Frames that are not 16:9 (including unreadable ones) are returned unchanged."""
    if img is None:
        return None
    h, w = img.shape[:2]
    if to_reference_size_factor(w, h) == 1.0:
        return img
    return cv2.resize(img, (REF_W, REF_H), interpolation=cv2.INTER_CUBIC)


def source_scale(img):
    """Pixels of the original frame / pixels of the frame scaled to 3840x2160 (1 for frames that to_reference_size does not upscale).
    Pixel values written by the engine (Sweep_X, Sweep_Start, etc.) are multiplied by this to return to original video pixels (the app places positions by the original video width)."""
    if img is None:
        return 1.0
    return to_reference_size_factor(img.shape[1], img.shape[0])


def to_reference_size_factor(w, h):
    """Ratio original / upscaled when to_reference_size scales a w x h frame to 3840x2160 (1 if not upscaled)."""
    if (w, h) == (REF_W, REF_H) or abs(w * REF_H - h * REF_W) > 0.01 * w * REF_H:
        return 1.0
    return w / REF_W


def reference_shape(img):
    """Shape of the frame after to_reference_size (3840x2160 if upscaled, otherwise unchanged)."""
    h, w = img.shape[:2]
    return (REF_H, REF_W) + img.shape[2:] if to_reference_size_factor(w, h) != 1.0 else img.shape


def crop_reference(img, x, y, w, h, margin=4):
    """The (x, y, w, h) region of the frame after to_reference_size. Instead of upscaling the whole frame, only that region is cut from the original frame and upscaled
    (Pass 1 reads every frame, so this saves the time to upscale the full 1280x720 frame to 4K).
    When the scale factor is an integer (3 for 1280x720, 2 for 1920x1080), cutting with a margin keeps the positions aligned, giving almost the same pixels
    as upscaling the whole frame and cutting (at most 1 level of difference on some pixels). Non-integer factors upscale the whole frame and cut."""
    sh, sw = img.shape[:2]
    if to_reference_size_factor(sw, sh) == 1.0:
        return img[y:y + h, x:x + w]
    k = REF_W / sw
    if abs(k - round(k)) > 1e-9 or abs(REF_H / sh - k) > 1e-9:
        return to_reference_size(img)[y:y + h, x:x + w]
    k = int(round(k))
    x0, y0 = max(0, x // k - margin), max(0, y // k - margin)
    x1, y1 = min(sw, -(-(x + w) // k) + margin), min(sh, -(-(y + h) // k) + margin)
    up = cv2.resize(img[y0:y1, x0:x1], ((x1 - x0) * k, (y1 - y0) * k), interpolation=cv2.INTER_CUBIC)
    return up[y - y0 * k:y - y0 * k + h, x - x0 * k:x - x0 * k + w]


def read_frame(frame_path):
    """Reads a frame image and scales it to 3840x2160 (to_reference_size). None if it cannot be read."""
    return to_reference_size(cv2.imread(frame_path))


def detect_sweep_x(frame_path):
    """
    Detects the X position of the sweep line in a frame image.

    Args:
        frame_path: str, path of the JPEG frame image

    Returns:
        int: start X coordinate of the sweep line (absolute). 0 if not detected.
    """
    return detect_sweep_x_img(read_frame(frame_path))


# Band containing the three waveform ROIs (HSV conversion is limited to this range)
_BAND_Y0 = min(ry for _, ry, _ in ROIS)
_BAND_Y1 = max(ry + rh for _, ry, rh in ROIS)


def detect_sweep_x_img(img):
    """Detects the X position of the sweep line in a frame (BGR) scaled to 3840x2160 (same as detect_sweep_x; 0 if not detected)."""
    if img is None or img.shape[0] < _BAND_Y1 or img.shape[1] < ROI_X + ROI_W:
        return 0
    return detect_sweep_x_band(img[_BAND_Y0:_BAND_Y1, ROI_X:ROI_X + ROI_W])


def detect_sweep_x_band(band):
    """Detects the X position of the sweep line in the band containing the three waveform ROIs (4K coordinates x = ROI_X to ROI_X+ROI_W, y = _BAND_Y0 to _BAND_Y1; BGR).
    HSV conversion covers only the ROI rows, and column runs are found with numpy (1.3.3; same result as the former detect_sweep_x_img)."""
    # Columns where "V_max excluding grid lines <= threshold" in all ROIs
    low_vmax = np.ones(ROI_W, dtype=bool)
    v_rois = []
    for name, ry, rh in ROIS:
        roi = cv2.cvtColor(band[ry - _BAND_Y0:ry - _BAND_Y0 + rh], cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]

        # Abnormal frame check: each ROI needs at least 100 waveform columns
        wave = (h_ch >= WAVE_H_MIN) & (h_ch <= WAVE_H_MAX) & (s_ch >= WAVE_S_MIN) & (v_ch >= WAVE_V_MIN)
        if np.count_nonzero(wave.any(axis=0)) < MIN_WAVE_COLS_PER_ROI:
            return 0

        # V_max excluding grid lines
        v_grid = np.where((h_ch >= GRID_H_MIN) & (h_ch <= GRID_H_MAX), 0, v_ch)
        v_rois.append(v_grid)
        low_vmax &= v_grid.max(axis=0) <= VMAX_THRESH

    # Find runs (gs to ge) of low V_max; keep those at least MIN_SWEEP_WIDTH wide with a column above the threshold on their left
    edges = np.diff(np.r_[0, low_vmax.astype(np.int8), 0])
    valid = []
    for gs, ge in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        if ge - gs + 1 < MIN_SWEEP_WIDTH or gs < 5:
            continue
        if low_vmax[max(0, gs - 5):gs].all():
            continue
        valid.append(int(gs))

    if not valid:
        return 0

    # Several candidates: choose the one with the highest V_max on its left (right after the active waveform)
    if len(valid) == 1:
        gs = valid[0]
    else:
        gs, best_left_v = None, 0
        for cand in valid:
            avg_left_v = np.mean([v[:, max(0, cand - 20):cand].max() for v in v_rois])
            if avg_left_v > best_left_v:
                best_left_v, gs = avg_left_v, cand

    return gs + ROI_X


# Constants for detecting the settings screen
_SETTINGS_CHECK_ROI_4K = (520, 269, 2877, 324)     # Pressure ROI (3840x2160)
_SETTINGS_CHECK_ROI_FHD = (260, 130, 1474, 186)    # Pressure ROI (1920x1080)
_SETTINGS_BRIGHT_V_THRESH = 50
_SETTINGS_BRIGHT_RATIO_THRESH = 0.3


def is_settings_screen(frame_path):
    """
    Detects whether the frame is a settings screen,
    from the V (brightness) distribution in the waveform ROI: a dialog
    makes the area unusually bright.

    Args:
        frame_path: str, path of the JPEG frame image

    Returns:
        bool: True for a settings screen
    """
    return is_settings_screen_img(read_frame(frame_path))


def is_settings_screen_img(img):
    """Same check as is_settings_screen, on a frame already read (scaled to 3840x2160)."""
    if img is None:
        return False

    h, w = img.shape[:2]
    rx, ry, rw, rh = _SETTINGS_CHECK_ROI_4K if w == 3840 else _SETTINGS_CHECK_ROI_FHD
    roi_bgr = img[ry:ry+rh, rx:rx+rw]
    # V = max(B, G, R), without HSV conversion
    v_channel = np.max(roi_bgr, axis=2)
    bright_ratio = np.mean(v_channel > _SETTINGS_BRIGHT_V_THRESH)
    return bright_ratio > _SETTINGS_BRIGHT_RATIO_THRESH


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_sweep.py <frame_path>")
        sys.exit(1)

    path = sys.argv[1]
    x = detect_sweep_x(path)
    print(x)


if __name__ == "__main__":
    main()
