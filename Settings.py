import os

# ============================================================
# Resolution-specific ROI coordinates
# Each resolution may have different camera positioning/zoom
# ============================================================

# Resolution format: (height, width) -> ROI definitions
ROI_BY_RESOLUTION = {
    (720, 1280): {  # 1280x720 (test.mov)
        "VTE":        [319, 19, 123, 47],
        "BreathMark": [61,  3,  99,  75],
        "Ppeak":      [143, 21, 87, 41],
    },
    (1080, 1920): {  # 1920x1080 (test2.mp4)
        "VTE":        [421, 28, 178, 75],
        "BreathMark": [36, 8, 131, 104],
        "Ppeak":      [215, 32, 130, 61],
    },
    (2160, 3840): {  # 3840x2160 (test3.mp4)
        "VTE":        [842, 56, 356, 150],
        "BreathMark": [0, 0, 344, 224],
        "Ppeak":      [430, 64, 260, 122],   # The PPEAK value at the top left (cmH2O), excluding the "PEAK" label above and the unit below. Used to estimate the waveform scale
    },
}

# Default (fallback) if resolution not in the map
DEFAULT_RESOLUTION = (720, 1280)
DEFAULT_ROIs = ROI_BY_RESOLUTION[DEFAULT_RESOLUTION]

# Base directory for reference images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# BreathMark reference images for inspiration phase identification.
# A/C/S × Start/End, pre-cropped to BreathMark ROI size.
# "Start" = inspiration onset, "End" = expiration onset.
BREATHMARK_REFERENCES = {
    name: os.path.join(BASE_DIR, f"{name}.jpg")
    for name in ["A_Start", "A_End", "C_Start", "C_End", "S_Start", "S_End"]
}

# Frames whose BreathMark ROI matches none of the reference images are marked Abnormal (grayscale template matching).
# Bennett980_Sample.mp4 (4K): 0.99 or higher with the icon, 0.53 when the icon is gone.
# 1280x720 captures from 2026-09 (upscaled to 4K before comparing): 0.75-0.86 with the icon
BREATHMARK_ABNORMAL_SCORE = 0.65

# Fallback reference for abnormal detection
REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "sample_frame.jpg")

# ROI regions for abnormal frame detection (x1, y1, x2, y2)
# TODO: Replace with actual coordinates
ROI_LIST_Abnormal_Detection = [
    (0, 0, 50, 400),
    (600, 0, 650, 400),
]
