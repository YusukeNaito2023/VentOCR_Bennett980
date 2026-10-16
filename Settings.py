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
        "Ppeak":      [430, 64, 260, 122],   # 左上の「PPEAK」の数値（cmH2O）。上の「PEAK」・下の単位は含めない。波形の目盛りの推定に使う
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

# BreathMark ROI が見本画像のどれとも似ていないフレームを Abnormal とする（ヒストグラム相関）。
# sample.mp4 では換気中は 0.94 以上、新患者設定画面・回路外れ（アイコンが消える）は 0.76〜0.81
BREATHMARK_ABNORMAL_SCORE = 0.85

# Fallback reference for abnormal detection
REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "sample_frame.jpg")

# ROI regions for abnormal frame detection (x1, y1, x2, y2)
# TODO: Replace with actual coordinates
ROI_LIST_Abnormal_Detection = [
    (0, 0, 50, 400),
    (600, 0, 650, 400),
]
