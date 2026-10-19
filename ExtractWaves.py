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

# 新しい画面（右端に届いた後、次の吸気で左端から描き直す画面）の最初の呼吸は、ROI左端から切り出し、
# trim_to_breath（_snap_to_breaths）で吸気の始まりへ合わせる。PB980 は新しい画面を吸気の約0.57秒前から描く。
# trim_to_breath が OFF のときだけ、描き始めの余白を秒で決めて ROI左端+余白から切り出す
# （sample.mp4（4K, 6秒/画面 = 479.5px/秒）で 206px。以前は ROI 幅の 7.16% で、1画面 24 秒の動画では 1.7 秒になり吸気が抜けていた）。
NEW_SCREEN_LEAD_IN_SEC = 0.43
# 開始フレームの sweep_x が ROI左端からこの時間以内で、前の呼吸と同じ画面でなければ、新しい画面の最初の呼吸とみなす
# （開始フレームではもう吸気が始まっているので、sweep_x から切り出すと立ち上がりの足元が欠ける）
NEW_SCREEN_MAX_SEC = 1.0

# Sweep line検出閾値（HSV）
# マーカーは明るい黄橙色の縦線としてPressure/Flow/Volume全ROIに出現
SWEEP_H_MIN = 20
SWEEP_H_MAX = 35
SWEEP_S_MIN = 150
SWEEP_V_MIN = 100
SWEEP_MIN_COLUMN_HITS = 20   # マーカーと判定するための最小ピクセル数/列
SWEEP_MAX_ROI_DEVIATION = 15  # ROI間のsweep line X位置の最大許容ずれ


def set_config(settings):
    global _config, SMOOTHING, IMPUTATION
    _config = settings
    SMOOTHING = settings.get("post_ocr", {}).get("smoothing", True)
    IMPUTATION = settings.get("post_ocr", {}).get("imputation", True)


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
    フレーム画像からsweep lineのX位置を検出する。

    Sweep lineは明るい黄橙色（H=20-35, S≥150, V≥100）の縦線として
    Pressure/Flow/Volume全ROIに一貫して出現する。

    Args:
        hsv_img: HSV画像（フルフレーム）
        rois: 波形ROI辞書 {"Pressure": (x,y,w,h), ...}

    Returns:
        int or None: 絶対画像座標のX位置。未検出時はNone。
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
            return None  # 1つのROIでも未検出なら失敗

        # 最もカウントが多い列をsweep line候補とする
        best_col = valid_cols[np.argmax(col_counts[valid_cols])]
        candidates.append(best_col + rx)  # 絶対座標に変換

    # ROI間のずれチェック
    if max(candidates) - min(candidates) > SWEEP_MAX_ROI_DEVIATION:
        return None

    return int(np.median(candidates))


# ============================================================
# Breath X range determination (sweep line + diff hybrid)
# ============================================================

def _determine_breath_x_range(img_start, img_end, hsv_end, rois):
    """
    1呼吸のアクティブ波形X範囲を決定する。

    戦略:
    1. 終了フレームのsweep lineマーカー → 右端
    2. 開始/終了フレームの差分 → 左端
    3. フォールバック: sweep line未検出時は差分で右端も決定

    Args:
        img_start: BGR画像（呼吸の開始フレーム）
        img_end: BGR画像（呼吸の終了フレーム）
        hsv_end: HSV画像（終了フレーム、事前変換済み）
        rois: 波形ROI辞書

    Returns:
        (x_left, x_right, is_fragmented, sweep_x): tuple
        - x_left: int, 波形左端の絶対X座標
        - x_right: int, 波形右端の絶対X座標
        - is_fragmented: bool, ROI右端に到達した呼吸
        - sweep_x: int or None, 検出されたsweep lineのX座標

        範囲決定不可の場合は (None, None, False, None)
    """
    sweep_x = detect_sweep_line(hsv_end, rois)
    diff_regions = get_difference_regions(img_end, img_start, rois)

    # ROI右端（Fragmented判定用）
    roi_right_edges = [rx + rw for rx, ry, rw, rh in rois.values()]
    max_right = max(roi_right_edges)

    if sweep_x is not None:
        # Sweep line検出成功: 右端 = sweep line
        x_right = sweep_x
        if diff_regions:
            x_left = min(r[0] for r in diff_regions)
        else:
            # 差分が検出されない場合はROI左端を使用
            x_left = max(rx for rx, ry, rw, rh in rois.values())
        is_fragmented = False
    elif diff_regions:
        # フォールバック: 差分のみ
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
    終了フレームの連続X範囲から波形を抽出する。

    Args:
        hsv_img: HSV画像（終了フレーム）
        rois: 波形ROI辞書
        x_left: int, 絶対X座標（開始、含む）
        x_right: int, 絶対X座標（終了、含む）

    Returns:
        dict {"Pressure": numpy array, "Flow": ..., "Volume": ...}
    """
    # 全ROIで共通のX範囲を計算（ROI間のrx差による長さ不一致を防ぐ）
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
# 縦軸の目盛り（ピクセル → 物理単位）
# ============================================================
# 画面上部の表示値と波形の画素から呼吸ごとに目盛りを逆算する（NAVA と同じ考え方）。
#   Pressure: PPEAK(cmH2O) ↔ 波形のピーク画素。0 の基線（青い横線）からの高さで比例させる
#   Volume:   VTE(mL) ↔ 波形のピーク画素（0 の基線から）
#   Flow:     表示値が無いので、Flow 画素の積分が Volume(mL) の増減に一致するように求める
# 求めた「最上段の目盛りの数字の値」をきりのよい値（NICE_STEPS）に丸め、前後の呼吸の最頻値で外れ値を除く。
# 寸法は 4K（980_sample.mp4、2026-10-11 実測）。他の解像度は高さの比で縮める。
#   最上段の目盛り: Pressure 40 → y289、Flow 120 → y734、Volume 1000 → y1175
#   0 の基線:       Pressure y516.5、Flow y904.5、Volume y1486.5（目盛りが変わると動きうるので呼吸ごとに検出する）
#   確認: PPEAK 18 ↔ ピーク 182px → 目盛り 38.8 → 40。VTE ↔ ピーク: 比 1.007（±4%）。Flow: 118.8（117.8〜119.3）→ 120
SCALE_UNITS = {"Pressure": "cmH2O", "Flow": "L/min", "Volume": "mL"}
SCALE_REF_HEIGHT = 2160
SCALE_TICK_Y = {"Pressure": 289, "Flow": 734, "Volume": 1175}
SCALE_ZERO_Y = {"Pressure": 516.5, "Flow": 904.5, "Volume": 1486.5}
ZERO_LINE_V = (28, 70)      # 0 の基線の明るさ（HSV の V）。背景は 10〜22
NICE_STEPS = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
NICE_TOLERANCE = 0.06       # 推定値ときりのよい値の差がこの割合以内なら丸める（それ以外は推定値をそのまま使う）
SCALE_SMOOTH = 9            # 前後この呼吸数の推定値の中央値で外れ値を除く
SCALE_MIN_VOTES = 3         # 前後に推定できた呼吸がこれ未満なら、その呼吸では決めない（近い呼吸の目盛りを使う）
SCALE_HYSTERESIS = 0.12     # 直前の目盛りとの差がこの割合未満なら目盛りは変わっていないとみなす
SWEEP_RATE_DEFAULT = 479.5  # 掃引の速さ px/秒（4K）。Sweep_Start/End から求められないとき用
FLOW_WINDOW = 12            # Volume の増減と Flow の積分を比べる区間の長さ（px）
# 呼吸の区切りと表示値の対応は1呼吸ずれることがある（980_sample では VTE は次の呼吸の行の値が波形と一致）。
# 呼吸ごとの「表示値 / 波形のピーク」のばらつきが最も小さいずれを自動で選ぶ
VALUE_SHIFTS = (0, -1)


def _zero_y(hsv_img, roi, name, ratio):
    """0 の基線（青い横線）の y。見つからなければ既定値。"""
    # 背景も同じ色相の暗い青（明るさ 10〜22）なので、明るさで分ける。ROI の上下の縁取り（明るい）は除く
    x, y, w, h = roi
    m = max(4, int(round(8 * ratio)))
    band = hsv_img[y + m:y + h - m, x:x + w]
    blue = (band[:, :, 0] >= 95) & (band[:, :, 0] <= 112) & (band[:, :, 1] >= 90) \
        & (band[:, :, 2] >= ZERO_LINE_V[0]) & (band[:, :, 2] <= ZERO_LINE_V[1])
    # 画面中央に出るポップアップ（「スタンバイ状態保留」など）の縁を拾わないよう、左端付近と右端付近の両方で
    # 線が見える行だけを基線の候補にする
    edge = max(1, int(w * 0.15))
    counts = np.minimum(blue[:, :edge].sum(axis=1), blue[:, -edge:].sum(axis=1))
    if counts.max() < 0.5 * edge:
        return SCALE_ZERO_Y[name] * ratio
    rows = np.flatnonzero(counts >= 0.5 * counts.max())
    rows = rows[np.abs(rows - rows[np.argmax(counts[rows])]) <= 3 * max(ratio, 0.5) * 2]   # 一番はっきりした線のまわりだけ
    found = y + m + float(rows.mean())
    # 最上段の目盛りより上（またはすぐ下）に基線があるはずはない: 画面の切り替わりなどの誤検出
    if found <= (SCALE_TICK_Y[name] + 40) * ratio:
        return SCALE_ZERO_Y[name] * ratio
    return found


def _peak_px(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 99)) if len(v) >= 20 else None


def _nice(estimate):
    """きりのよい値に丸める（遠ければ有効数字2桁で丸めるだけ）。"""
    if estimate is None or not np.isfinite(estimate) or estimate <= 0:
        return None
    exp = 10.0 ** np.floor(np.log10(estimate))
    cands = [s * exp for s in NICE_STEPS] + [10 * exp]
    best = min(cands, key=lambda c: abs(c - estimate))
    if abs(best - estimate) <= NICE_TOLERANCE * best:
        return float(best)
    return float(round(estimate, -int(np.floor(np.log10(estimate))) + 1))


def _smooth_tops(estimates, width=SCALE_SMOOTH):
    """呼吸順の推定値（None 可）を、前後 width 呼吸の中央値にしてから、きりのよい値に丸める。
    1呼吸ごとの推定は ±15% ほどばらつくので、直前の目盛りとの差が SCALE_HYSTERESIS 未満なら直前の目盛りのままにする。
    推定できない呼吸（前後の推定が SCALE_MIN_VOTES 未満）は近い呼吸の値で埋める。"""
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
    """表示値 values（呼吸順）と波形のピークから、最上段の目盛りの値の推定（呼吸順、None 可）。"""
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
    """VALUE_SHIFTS のうち推定値のばらつき（対数の四分位範囲）が最も小さいずれで推定する。"""
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
    """Flow の積分 = Volume の増減 となるように、Flow の最上段の目盛りの値を求める。"""
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
        if min(a, b) < gv["zero"] + 4 or max(a, b) > gv["h"] - 6:   # 0 付近・上端に張り付いた Volume は使わない
            continue
        xs.append(seg.sum() / rate * 1000.0 / 60.0)
        ys.append((b - a) * ml_per_px)
    if r["partial"] or len(xs) < 30:
        return None
    xs, ys = np.array(xs), np.array(ys)
    sxx = float(xs @ xs)
    return float(xs @ ys) / sxx * (gf["tick"] - gf["zero"]) if sxx > 0 else None


def _scale(name, top_value, g):
    """最上段の目盛りの値 → 波形CSVのピクセル値の範囲 [0, h] に対応する {min, max}。"""
    k = top_value / (g["tick"] - g["zero"])    # 単位 / px
    return {"min": round(-g["zero"] * k, 2), "max": round((g["h"] - g["zero"]) * k, 2), "top": top_value,
            "unit": SCALE_UNITS[name]}


def estimate_scales(records, rate):
    """{BreathNumber: {name: scale}}。records: 呼吸順の dict（breath, waves, geom, ppeak, vte, partial）
    partial（右端で見切れた・異常フレームを含む呼吸）は推定に使わず、前後の呼吸の目盛りを使う。"""
    p_tops = _smooth_tops(_aligned_estimates(records, "Pressure", "ppeak"))
    v_tops = _smooth_tops(_aligned_estimates(records, "Volume", "vte"))
    f_tops = _smooth_tops([_flow_top(r, vt, rate) if vt else None for r, vt in zip(records, v_tops)])
    out = {}
    for r, pt, ft, vt in zip(records, p_tops, f_tops, v_tops):
        out[r["breath"]] = {nm: _scale(nm, t, r["geom"][nm])
                            for nm, t in (("Pressure", pt), ("Flow", ft), ("Volume", vt)) if t}
    return out


def _to_physical(values, scale, h):
    v = np.asarray(values, dtype=float)
    return np.round(scale["min"] + v / h * (scale["max"] - scale["min"]), 2)


# ============================================================
# 呼吸の区切りを吸気の始まりへ合わせる（NAVA v1.3 の切り詰めと同じ考え方）
# ============================================================
# 1呼吸 = 吸気の始まり 〜 次の吸気の始まり。
# 波形が時間の上で隙間なくつながっている呼吸どうしは、区切りを「近くの吸気の始まりの TRIM_LEAD_SEC 前」へ動かし、
# 行を前後の呼吸の間で移す（データは消えない）。つながりの端（前後の呼吸に波形が無い）は、
# 吸気の始まりより前（前の呼吸の呼気の終わり）と、呼気を伴わない最後の吸気（次の呼吸の吸気の途中）を削る。
ONSET_FLOW = 4.0        # Flow がこれ（L/min）を超えたら吸気の流れ
ONSET_RUN_SEC = 0.06    # 吸気の流れがこれ以上続くこと
ONSET_PEAK = 12.0       # その吸気がこれ（L/min）以上に届くこと
QUIET_SEC = 0.06        # 吸気の前に、吸気でない区間がこれ以上あること
ONSET_FRAC = 0.2        # 吸気の最大値のこの割合を超えた点から、立ち上がりの足元までさかのぼる
EXP_FLOW = 4.0          # 呼気の流れ（L/min, 0 よりこれ以上下）
BACKTRACK_TOL = 0.3     # 足元までさかのぼるとき、これ（L/min）以上下がり続ける間はさかのぼる
TRIM_LEAD_SEC = 0.04    # 吸気の始まりのこの時間前から残す
CONTIG_TOL_SEC = 0.15   # 前の呼吸の最後の点の次の時刻と、次の呼吸の最初の点の時刻の差がこれ（秒）以内ならつながっている（時刻の推定の誤差を見込む）


def _inspiration_onsets(flow, rate):
    """flow: Flow（L/min, NaN 可）, rate: 点/秒。吸気の始まり（立ち上がりの足元）の行番号の一覧。"""
    raw = np.asarray(flow, dtype=float)
    g = pd.Series(raw).interpolate(limit_area="inside").to_numpy()
    n = len(g)
    fin = np.isfinite(g)
    ins = fin & (np.nan_to_num(g) > ONSET_FLOW)
    run_min = max(2, int(round(ONSET_RUN_SEC * rate)))
    quiet_min = max(2, int(round(QUIET_SEC * rate)))
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
        while k >= 0 and fin[k] and not ins[k]:
            quiet += 1
            k -= 1
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
    """items: 時間順の dict のリスト {"wave": DataFrame（Flow 列, 行=点）または None, "t0": 秒, "rate": 点/秒}。
    各 item に "wave"（切り直した波形）, "t0", "rate", "trim_head", "trim_tail"（元の範囲から削った/渡した点の数。負は前後から受け取った点）を書き込む。"""
    # つながっている呼吸のまとまりに分ける
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
        # 先頭: 空の行と、最初の呼吸の吸気の始まりより前
        lo = int(np.argmax(valid)) if valid.any() else 0
        first = on[on < (bounds[0] + bounds[1]) // 2]   # 最初の呼吸の前半にある吸気の始まり
        new[0] = max(lo, int(first[0]) - lead) if len(first) else lo
        # 呼吸の間の区切り: 前後の呼吸の真ん中より内側にある吸気の始まりのうち、元の区切りに最も近いもの
        for j in range(1, len(g)):
            a = (bounds[j - 1] + bounds[j]) // 2
            b = (bounds[j] + bounds[j + 1]) // 2
            cand = on[(on > a) & (on < b)]
            if len(cand):
                o = int(cand[np.argmin(np.abs(cand - bounds[j]))])
                new[j] = o - lead
        # 末尾: 空の行と、呼気を伴わない最後の吸気（次の呼吸の吸気の途中）
        hi = len(cat) - int(np.argmax(valid[::-1])) if valid.any() else len(cat)
        new[-1] = hi
        last_on = on[(on > new[-2] + lead) & (on < hi)]
        if len(last_on) and not (np.nan_to_num(flow[last_on[-1]:hi]) < -EXP_FLOW).any():
            new[-1] = max(new[-2] + 1, int(last_on[-1]) - lead)
        # 順序を保つ（区切りが追い越さない）
        for j in range(1, len(new)):
            new[j] = max(new[j], new[j - 1] + 1)
        for j, i in enumerate(g):
            a, b = int(new[j]), int(new[j + 1])
            items[i]["trim_head"] = a - int(bounds[j])
            items[i]["trim_tail"] = int(bounds[j + 1]) - b
            items[i]["wave"] = cat.iloc[a:b].reset_index(drop=True)
            items[i]["t0"] = float(times[a])
            # 前後の呼吸から受け取った点は元の呼吸の速さで時刻が決まっているので、速さは両端の点の時刻から求め直す
            if b - a >= 2 and times[b - 1] > times[a]:
                items[i]["rate"] = float((b - a - 1) / (times[b - 1] - times[a]))
    return items


def _sweep_rate_from_frames(frames, fps, src_scale):
    """Frames.csv の Sweep_X（元の動画の画素）の、隣のフレームとの進み方の中央値から掃引の速さ（4K の px/秒）。求められなければ None。"""
    if "Sweep_X" not in frames.columns:
        return None
    f = frames[["Frame", "Sweep_X"]].apply(pd.to_numeric, errors="coerce").sort_values("Frame")
    dx, dfr = f["Sweep_X"].diff().to_numpy(), f["Frame"].diff().to_numpy()
    ok = np.isfinite(dx) & (dfr == 1) & (dx > 0)   # 折り返し（負）と検出できなかったフレームは使わない
    if ok.sum() < 10:
        return None
    return float(np.median(dx[ok])) * fps / src_scale


def _to_source_px(df, scale):
    """4K の画素ごとの行の波形を、元の動画の画素ごとの行に間引く（scale = 元 / 4K。1 ならそのまま）。
    拡大したフレームの行は補間で作ったもので、元の動画で見分けられるのは元の画素の単位まで。"""
    if scale >= 1.0 or len(df) == 0:
        return df
    n = max(1, int(np.ceil(len(df) * scale)))
    idx = np.minimum(np.round(np.arange(n) / scale).astype(int), len(df) - 1)
    return df.iloc[idx].reset_index(drop=True)


def BreathsToCSVs(breaths, frames, frames_root, output_dir, user_id, overwrite=False):
    """
    捜査線ベースで1呼吸分の波形CSVを生成する。

    開始/終了フレームの捜査線位置で4パターンに分類:
      パターン1: 開始=0, 終了=0 → 異常フレーム（Abnormal）、スキップ
      パターン2: 開始>0, 終了=0 → 右端で見切れ（Fragmented）。1.3.0: 開始はパターン4と同じく前の呼吸の続きから
      パターン3: 開始=0, 終了>0 → 新しい画面の最初の呼吸。ROI左端（trim_to_breath が OFF なら＋描き始めの余白
                 NEW_SCREEN_LEAD_IN_SEC）～終了sweep_xを「終了フレーム」で解析。1.3.2: 開始の sweep_x が左端の近くにある
                 新しい画面の最初の呼吸（パターン4）も同じ左端から
      パターン4: 開始>0, 終了>0 → 開始sweep_x～終了sweep_xを「終了フレーム」で解析
                 1.3.0: 開始は「前の呼吸の最後のフレームの sweep_x の次の列」から（以前は開始フレームの sweep_x からで、
                 呼吸の間の 1 フレーム分（約 0.1 秒）の波形がどの呼吸にも入っていなかった）

    1.3.0: 各呼吸の最初の点の時刻 Wave_T0（秒）と Wave_Rate（px/秒）を書き出し、trim_to_breath（既定 ON）なら
    波形の区切りを吸気の始まりへ合わせる（_snap_to_breaths。点を前後の呼吸の間で移す）。X は 1 からの行番号。

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
    from .detect_sweep import detect_sweep_x_img, is_settings_screen_img, read_frame, to_reference_size, source_scale

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
    records = []       # 目盛りの推定用: 呼吸順の画素波形と寸法（推定後に物理単位へ換算して書き出す）
    sweep_rates = []
    fps = float((_config or {}).get("post_ocr", {}).get("fps", 10))
    values = breaths.set_index("BreathNumber")

    # source_file → サブフォルダ名のマッピングを構築
    def _subfolder_from_source(source_file):
        import re
        m = re.match(r"output_(\d{5})\.csv", str(source_file))
        return m.group(1) if m else None

    # ROI左端（パターン3で使用）
    sample_path = os.path.join(frames_root, f"frame_{int(frames['Frame'].iloc[0]):08d}.jpg")
    sample_raw = cv2.imread(sample_path)
    sample_img = to_reference_size(sample_raw)
    # 元の動画の画素 / 4K の画素。書き出す画素の値（Sweep・Region・Wave_Rate・波形の行）は元の動画の画素に戻す
    src_scale = source_scale(sample_raw)
    if sample_img is not None:
        rois = _get_waveform_rois(sample_img.shape)
        roi_x_left = max(rx for rx, ry, rw, rh in rois.values())
        roi_x_right = min(rx + rw for rx, ry, rw, rh in rois.values())
    else:
        roi_x_left = 520
        roi_x_right = 520 + 2877

    # 新しい画面の最初の呼吸の切り出しに使う掃引の速さ（4K の px/秒）。Pass 1 のフレームごとの Sweep_X の進み方から
    early_rate = _sweep_rate_from_frames(frames, fps, src_scale) or SWEEP_RATE_DEFAULT
    trim_on = (_config or {}).get("post_ocr", {}).get("trim_to_breath", True)
    new_screen_left = roi_x_left if trim_on else roi_x_left + int(round(NEW_SCREEN_LEAD_IN_SEC * early_rate))

    # 読んだフレームを少しだけ覚えておく（1.3.3）。終了フレームは掃引線の検出・設定画面の判定・波形の抽出で、
    # 次の呼吸では「開始フレームの1つ前」として使うので、同じ画像を何度も読み直さない
    frame_cache = {}

    def _frame(path):
        if path not in frame_cache:
            if len(frame_cache) >= 4:
                frame_cache.pop(next(iter(frame_cache)))
            frame_cache[path] = read_frame(path)
        return frame_cache[path]

    # 波形データなしの場合に空CSVを書き出すヘルパー
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

        # Abnormalフレームを含む呼吸はフラグを立てる
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

        # 開始/終了フレームの捜査線位置を検出
        sweep_start = detect_sweep_x_img(_frame(start_path))
        sweep_end = detect_sweep_x_img(_frame(end_path))

        def _previous_sweep():
            """前の呼吸の最後のフレーム（= 開始フレームの1つ前）の sweep_x。前の呼吸で検出済みの値を使う。"""
            prev_end = sweep_end_map.get(breath_numbers[i - 1]) if i > 0 and prev_end_frame == start_frame_num - 1 else None
            if prev_end is None:
                prev_path = os.path.join(frames_root, f"frame_{start_frame_num - 1:08d}.jpg")
                prev_end = detect_sweep_x_img(_frame(prev_path)) if os.path.isfile(prev_path) else 0  # engine_api 3 では isfile で確かめる
            return prev_end

        def _left_from_previous(x_left):
            """前の呼吸の最後のフレームの sweep_x の次の列（同じ画面のときだけ）。呼吸の間の 1 フレーム分の抜けを無くす。
            新しい画面の最初の呼吸（開始フレームの sweep_x が左端の近くで、前の呼吸と同じ画面でない）は new_screen_left から。"""
            prev_end = _previous_sweep()
            if 0 < prev_end < x_left:
                return prev_end + 1
            if x_left - roi_x_left <= NEW_SCREEN_MAX_SEC * early_rate:
                return min(new_screen_left, x_left)
            return x_left

        sweep_start_map[bn] = sweep_start
        sweep_end_map[bn] = sweep_end

        # 1個目のBreathはFragmentedとして扱う（開始位置が不明）
        if is_first_breath:
            if sweep_end > 0:
                x_left = roi_x_left
                x_right = sweep_end
            else:
                x_left = roi_x_left
                x_right = roi_x_right
            fragmented_flags[bn] = True
            alert_reasons.append("Fragment")
        # パターン判定
        elif sweep_start == 0 and sweep_end == 0:
            # パターン1: 両方検出不可 — 設定変更画面かNo Sweepかを判別
            if is_settings_screen_img(_frame(end_path)):
                alert_reasons.append("Settings Change")
            else:
                alert_reasons.append("No Sweep")
            _write_empty_csv(csv_path)
            effective += 1
            alert_map[bn] = ", ".join(alert_reasons)
            continue
        elif sweep_start > 0 and sweep_end == 0:
            # パターン2: 右端で見切れ（Fragmented）→ sweep_start～ROI右端を解析
            x_left = _left_from_previous(sweep_start)
            x_right = roi_x_right
            fragmented_flags[bn] = True
            alert_reasons.append("Fragment")
        elif sweep_start == 0 and sweep_end > 0:
            # パターン3: 新しい画面の最初の呼吸。ROI左端（trim_to_breath が OFF なら＋描き始めの余白）～終了sweep_xを終了フレームで解析
            x_left = new_screen_left
            if x_left >= sweep_end:
                x_left = roi_x_left
            x_right = sweep_end
            fragmented_flags[bn] = False
        else:
            # パターン4: 開始sweep_x～終了sweep_xを終了フレームで解析
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

        # 終了フレームから波形を抽出
        img_end = _frame(end_path)
        if img_end is None:
            alert_reasons.append("Frame Read Error")
            _write_empty_csv(csv_path)
            effective += 1
            alert_map[bn] = ", ".join(alert_reasons)
            continue

        rois = _get_waveform_rois(img_end.shape)
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

        # 目盛りの推定に使う寸法（ピクセル値の座標 = ROI 下端からの画素数）
        ratio = img_end.shape[0] / SCALE_REF_HEIGHT
        geom = {}
        for name, (rx, ry, rw, rh) in rois.items():
            bottom = ry + rh
            geom[name] = {"h": rh, "zero": bottom - _zero_y(hsv_end, (rx, ry, rw, rh), name, ratio),
                          "tick": bottom - SCALE_TICK_Y[name] * ratio}
        # 最初の点の時刻を求めるための位置（_extract_waveforms_in_range と同じ切り詰め）
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

    # 目盛りを推定して物理単位へ換算し、書き出す
    rate = float(np.median(sweep_rates)) if len(sweep_rates) >= 3 else SWEEP_RATE_DEFAULT
    scales = estimate_scales(records, rate) if records else {}
    for rec in records:
        sc = scales.get(rec["breath"], {})
        for nm, scale in sc.items():
            rec["df"][nm] = _to_physical(rec["df"][nm], scale, rec["geom"][nm]["h"])

    # 最初の点の時刻: 終了フレームの sweep_x（ペン先）から逆算（アプリの終了側基準と同じ）。
    # 終了フレームで sweep_x が無い（見切れ）呼吸は開始フレームの sweep_x から
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

    # 波形の区切りを吸気の始まりへ合わせる（Flow の目盛りが求まっていない呼吸は対象外）
    items = [{"wave": rec["df"] if "Flow" in scales.get(rec["breath"], {}) else None,
              "t0": wave_time[rec["breath"]], "rate": rate} for rec in records]
    if (_config or {}).get("post_ocr", {}).get("trim_to_breath", True):
        _snap_to_breaths(items)
    wave_info = {}
    for rec, it in zip(records, items):
        df = rec["df"] if it["wave"] is None else it["wave"]
        t0 = wave_time[rec["breath"]] if it["wave"] is None else it["t0"]
        r = rate if it["wave"] is None else it["rate"]
        df = _to_source_px(df, src_scale).copy()   # 行・速さ・動かした点の数は元の動画の画素で
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

    # Breaths DataFrameにメタデータカラムを追加
    breaths["Fragmented"] = breaths["BreathNumber"].map(fragmented_flags).fillna(False).astype(bool)
    # 画面の x は元の動画の画素で書く（アプリはキャプチャーの走査線などを元の動画の幅で出す）
    def _src_px(m):
        return {b: (x if src_scale == 1.0 else round(x * src_scale, 1)) for b, x in m.items()}
    breaths["Region_X_Start"] = breaths["BreathNumber"].map(_src_px(region_x_start_map)).fillna("")
    breaths["Region_X_End"] = breaths["BreathNumber"].map(_src_px(region_x_end_map)).fillna("")
    breaths["Sweep_Start"] = breaths["BreathNumber"].map(_src_px(sweep_start_map)).fillna("")
    breaths["Sweep_End"] = breaths["BreathNumber"].map(_src_px(sweep_end_map)).fillna("")

    # Alert列を設定
    breaths["Alert"] = breaths["BreathNumber"].map(alert_map).fillna("")

    # Start_File_Name / End_File_Name を生成
    # Label画面が "subfolder/frame_XXXXXXXX.jpg" 形式を期待するため、
    # iFrame（ローカルフレーム番号）とsource_file（出力CSV名）から構築
    start_file_names = {}
    end_file_names = {}
    for bn in breath_numbers:
        breath_frames = frames[frames["BreathNumber"] == bn]
        if breath_frames.empty:
            continue
        # 開始フレーム行
        start_row = breath_frames.loc[breath_frames["Frame"].idxmin()]
        subfolder = _subfolder_from_source(start_row.get("source_file", ""))
        iframe = int(start_row.get("iFrame", start_row["Frame"]))
        if subfolder:
            start_file_names[bn] = f"{subfolder}/frame_{iframe:08d}.jpg"

        # 終了フレーム行
        end_row = breath_frames.loc[breath_frames["Frame"].idxmax()]
        subfolder = _subfolder_from_source(end_row.get("source_file", ""))
        iframe = int(end_row.get("iFrame", end_row["Frame"]))
        if subfolder:
            end_file_names[bn] = f"{subfolder}/frame_{iframe:08d}.jpg"

    breaths["Start_File_Name"] = breaths["BreathNumber"].map(start_file_names).fillna("")
    breaths["End_File_Name"] = breaths["BreathNumber"].map(end_file_names).fillna("")

    return effective, breaths
