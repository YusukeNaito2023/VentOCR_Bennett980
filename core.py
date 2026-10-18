"""
Bennett 980 VentOCR Engine
VTE（一回換気量）とBreathMark（呼吸境界のカラー反転検出）を解析する。

2パス方式:
- Pass 1: BreathMark検出（全フレーム、カラーヒストグラム比較）
- Pass 2: VTE + BreathMark_char OCR（BreathMark間の中間フレームのみ）

Interface:
    initial_ocr(frames, total_frames, subfolder, settings, progress_callback, frames_dir)
        -> pd.DataFrame (1 row per frame)

    post_ocr(df, Analyze_Id, user_ID, settings)
        -> (df_frames, df_breaths)

    extract_waves(df_frames, df_breaths, frames_root, output_dir, user_ID, settings, overwrite)
        -> (effective_breaths_count, df_breaths)
"""

from tqdm import tqdm
from easyocr import Reader
from . import Settings
from . import ExtractWaves as ew
from .detect_sweep import read_frame, to_reference_size, source_scale, detect_sweep_x_img
import cv2
import re
import pandas as pd
import numpy as np
import os

ENGINE_VERSION = "1.3.2"
VENTOCR_TESTED = "0.5.5"


# ============================================================
# EasyOCR initialization
# ============================================================

_EASYOCR_LOCKFILE = "easyocr_init.lock"


def _get_easyocr_model_dir():
    model_dir = os.path.join(".", "_runtime_models", "easyocr")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def _easyocr_model_ready(model_dir):
    from pathlib import Path
    p = Path(model_dir)
    if not p.exists():
        return False
    patterns = ["craft*.pth", "dbnet*.pth", "latin*.pth", "english_g2*.pth", "standard*.pth"]
    for pat in patterns:
        if list(p.glob(pat)):
            return True
    return False


def _acquire_easyocr_file_lock(model_dir, timeout=300, poll=0.5):
    import time
    lock_path = os.path.join(model_dir, _EASYOCR_LOCKFILE)
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd, lock_path
        except FileExistsError:
            if (time.time() - start) > timeout:
                raise TimeoutError(f"Timeout waiting for EasyOCR lock: {lock_path}")
            time.sleep(poll)


def _release_easyocr_file_lock(fd, lock_path):
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def _ensure_easyocr_reader_ready(gpu_enabled):
    model_dir = _get_easyocr_model_dir()
    if _easyocr_model_ready(model_dir):
        return Reader(["en"], gpu=gpu_enabled, model_storage_directory=model_dir, download_enabled=True)

    fd, lock_path = _acquire_easyocr_file_lock(model_dir)
    try:
        if not _easyocr_model_ready(model_dir):
            Reader(["en"], gpu=gpu_enabled, model_storage_directory=model_dir, download_enabled=True)
    finally:
        _release_easyocr_file_lock(fd, lock_path)

    return Reader(["en"], gpu=gpu_enabled, model_storage_directory=model_dir, download_enabled=True)


# ============================================================
# OCR result parsing
# ============================================================

def _parse_ocr_result(result):
    raw_texts = []
    confidences = []

    for res in result:
        _, text, score = res
        text = text.replace("_", ".").replace("I", "1")
        raw_texts.append(text)
        confidences.append(score)

    full_text = ", ".join(raw_texts)
    confidence_str = ", ".join([f"{s:.2f}" for s in confidences])

    numeric_parts = [re.sub(r"[^0-9.:]", "", t) for t in raw_texts]
    numeric_parts = [t for t in numeric_parts if re.match(r"^\d+\.?\d*$", t)]

    if len(numeric_parts) == 1:
        cleaned_value = numeric_parts[0]
    elif len(numeric_parts) >= 2:
        if numeric_parts[0].endswith('.'):
            cleaned_value = numeric_parts[0] + numeric_parts[1]
        else:
            cleaned_value = numeric_parts[0] + '.' + numeric_parts[1]
    else:
        cleaned_value = ""

    return cleaned_value, full_text, confidence_str


# ============================================================
# ROI resolution detection and selection
# ============================================================

def _get_roi_for_frame(key, frame_shape):
    """Get ROI coordinates for a specific key, matching frame resolution."""
    frame_h, frame_w = frame_shape[:2]
    frame_res = (frame_h, frame_w)

    if frame_res in Settings.ROI_BY_RESOLUTION:
        return Settings.ROI_BY_RESOLUTION[frame_res].get(key)

    return Settings.DEFAULT_ROIs.get(key)


# ============================================================
# Two-pass initial_ocr
# ============================================================

def _load_color_frame(frames_dir, frame_index):
    """Load a color frame from disk by index."""
    path = os.path.join(frames_dir, f"frame_{frame_index:08d}.jpg")
    return read_frame(path)


def initial_ocr(frames, total_frames, subfolder, settings={},
                progress_callback=None, frames_dir=None):
    """
    2パスOCR方式:
    - Pass 1: BreathMark遷移検出（全フレーム、カラージェネレータから順次受け取り）
    - Pass 2: VTE + BreathMark_char OCR（BreathMark間の中間フレーム、frames_dirからディスク読み込み）

    Args:
        frames: カラーフレームのジェネレータ (BGR, numpy配列)。list化禁止。
        total_frames: int
        subfolder: str
        settings: dict
        progress_callback: callable or None
        frames_dir: str フレームJPEG保存ディレクトリ（ランダムアクセス用）

    Returns:
        pd.DataFrame with columns: Frame, VTE, VTE_raw, VTE_conf, BreathMark, BreathMark_raw, BreathMark_char
    """

    ocr_settings = settings.get("initial_ocr", {})
    gpu_enabled = ocr_settings.get("use_gpu", True)

    reader = _ensure_easyocr_reader_ready(gpu_enabled)
    print(f"GPU:{gpu_enabled}")

    print(f"\n[Pass 1] Detecting BreathMarks (all {total_frames} frames)...")

    # ============================================================
    # PASS 1: BreathMark detection
    # ============================================================

    data = []
    prev_bm_best = None
    frame_shape = None
    detected_resolution = None
    bm_frame_indices = []  # Track which frames have BreathMark=1

    # BreathMark phase identification via grayscale template matching (TM_CCOEFF_NORMED).
    # 色ヒストグラムは取り込み機器による色合いの違い（2026-09 の 1280x720 の動画など）で見本と一致しなくなるため、
    # 明るさ・コントラストの違いに強い正規化相互相関で、A/C/S Start/End の見本の形と比べる。
    _ref_grays = {}  # {"A_Start": gray image, "A_End": gray image, ...}
    _bm_roi_size = None  # (w, h) for resizing references

    with tqdm(total=total_frames, desc="Pass 1: BreathMark Detection") as pbar:
        if progress_callback is not None:
            progress_callback(current=0, total=total_frames, message="Detecting BreathMarks")

        for frame_count, color_frame in enumerate(frames):
            # Initialize row
            row = {
                "Frame": frame_count,
                "VTE": "",
                "VTE_raw": "",
                "VTE_conf": "0.00",
                "Ppeak": "",
                "Ppeak_raw": "",
                "Ppeak_conf": "0.00",
                "BreathMark": 0,
                "BreathMark_raw": "",
                "BreathMark_char": "",
                "Abnormal": 0,
                "BM_Best": "",
                "BM_BestScore": 0.0,
                "BM_A_Start": 0.0,
                "BM_A_End": 0.0,
                "BM_C_Start": 0.0,
                "BM_C_End": 0.0,
                "BM_S_Start": 0.0,
                "BM_S_End": 0.0,
                "Sweep_X": np.nan,  # ペン先（捜査線）の x（元の動画の画素）。アプリのキャプチャーの走査線に使う。検出できなければ空
            }

            scale = source_scale(color_frame)
            color_frame = to_reference_size(color_frame)  # 1280x720 などは 3840x2160 に拡大して 4K の座標で読む
            if color_frame is not None:
                sweep_x = detect_sweep_x_img(color_frame)
                if sweep_x > 0:
                    row["Sweep_X"] = round(sweep_x * scale, 1)

                if frame_shape is None:
                    frame_shape = color_frame.shape
                    detected_resolution = (frame_shape[0], frame_shape[1])
                    print(f"[info] Detected resolution: {detected_resolution[1]}x{detected_resolution[0]}")

                    # Load BreathMark references (grayscale, resized to the ROI)
                    bm_roi_def = _get_roi_for_frame("BreathMark", frame_shape)
                    if bm_roi_def is not None:
                        _bm_roi_size = (bm_roi_def[2], bm_roi_def[3])  # (w, h)
                    for name, ref_path in Settings.BREATHMARK_REFERENCES.items():
                        if os.path.exists(ref_path):
                            ref_img = cv2.imread(ref_path)
                            if _bm_roi_size is not None:
                                ref_img = cv2.resize(ref_img, _bm_roi_size)
                            _ref_grays[name] = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    print(f"[BreathMark] Loaded {len(_ref_grays)} reference images: {list(_ref_grays.keys())}")

                bm_roi = _get_roi_for_frame("BreathMark", color_frame.shape)
                if bm_roi is not None:
                    bx, by, bw, bh = bm_roi
                    bm_color_roi = color_frame[by:by+bh, bx:bx+bw]

                    # 毎フレーム: A/C/S_Start/End 6枚全てとテンプレートマッチング
                    if _ref_grays:
                        roi_gray = cv2.cvtColor(bm_color_roi, cv2.COLOR_BGR2GRAY)
                        best_name = ""
                        best_score = -1.0
                        for name, ref_gray in _ref_grays.items():
                            score = float(cv2.matchTemplate(roi_gray, ref_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
                            row[f"BM_{name}"] = round(score, 4)
                            if score > best_score:
                                best_score = score
                                best_name = name
                        row["BM_Best"] = best_name
                        row["BM_BestScore"] = round(best_score, 4)
                        if best_score < Settings.BREATHMARK_ABNORMAL_SCORE:
                            row["Abnormal"] = 1

                        # BreathMark検出: _End → _Start 遷移で吸気開始を判定
                        if (prev_bm_best is not None
                                and "_End" in prev_bm_best
                                and "_Start" in best_name):
                            row["BreathMark"] = 1
                            row["BreathMark_raw"] = f"{prev_bm_best} -> {best_name} (score={best_score:.4f})"
                            bm_frame_indices.append(frame_count)

                        prev_bm_best = best_name

            data.append(row)
            pbar.update(1)

            if progress_callback is not None and (frame_count % 100 == 0 or frame_count == total_frames - 1):
                progress_callback(current=frame_count, total=total_frames, message="Detecting BreathMarks")

    df = pd.DataFrame(data)

    print(f"[Pass 1] Complete: {len(bm_frame_indices)} BreathMarks detected\n")

    # ============================================================
    # PASS 2: VTE + BreathMark_char OCR at midpoints
    # ============================================================

    print(f"[Pass 2] OCR at midpoints ({len(bm_frame_indices)-1} midpoints)...")

    with tqdm(total=len(bm_frame_indices) - 1, desc="Pass 2: Midpoint OCR") as pbar:
        for i in range(len(bm_frame_indices) - 1):
            bm1 = bm_frame_indices[i]
            bm2 = bm_frame_indices[i + 1]
            midpoint = (bm1 + bm2) // 2

            if progress_callback is not None:
                progress_callback(current=i, total=len(bm_frame_indices) - 1, message="OCR at midpoints")

            # OCR VTE: frames_dirからカラーJPEGを読み込み、グレースケール変換してOCR
            mid_color_frame = _load_color_frame(frames_dir, midpoint) if frames_dir is not None else None
            if mid_color_frame is not None:
                gray_frame = np.dot(mid_color_frame[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
                vte_roi = _get_roi_for_frame("VTE", gray_frame.shape)
                if vte_roi is not None:
                    x, y, w, h = vte_roi
                    roi = gray_frame[y:y+h, x:x+w]
                    try:
                        vte_result = reader.readtext(roi)
                        vte_value, vte_raw, vte_conf = _parse_ocr_result(vte_result)
                        df.loc[df["Frame"] == bm1, "VTE"] = vte_value
                        df.loc[df["Frame"] == bm1, "VTE_raw"] = vte_raw
                        df.loc[df["Frame"] == bm1, "VTE_conf"] = vte_conf
                    except Exception:
                        pass

            # OCR Ppeak: 呼吸の終わり（次の BreathMark のフレーム）の表示がこの呼吸の値（吸気が終わると更新される）
            end_color_frame = _load_color_frame(frames_dir, bm2) if frames_dir is not None else None
            if end_color_frame is not None:
                pp_roi = _get_roi_for_frame("Ppeak", end_color_frame.shape)
                if pp_roi is not None:
                    x, y, w, h = pp_roi
                    gray = cv2.cvtColor(end_color_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    try:
                        # 大きな数値だけを採る（枠に入った見出しや単位の文字の切れ端を数字と読むことがある）
                        res = reader.readtext(gray, allowlist="0123456789")
                        res = sorted(res, key=lambda r: -(max(p[1] for p in r[0]) - min(p[1] for p in r[0])))[:1]
                        pp_value, pp_raw, pp_conf = _parse_ocr_result(res)
                        df.loc[df["Frame"] == bm1, "Ppeak"] = pp_value
                        df.loc[df["Frame"] == bm1, "Ppeak_raw"] = pp_raw
                        df.loc[df["Frame"] == bm1, "Ppeak_conf"] = pp_conf
                    except Exception:
                        pass

            # OCR BreathMark character from color midpoint frame
            color_frame = mid_color_frame
            if color_frame is not None:
                bm_roi = _get_roi_for_frame("BreathMark", color_frame.shape)
                if bm_roi is not None:
                    bx, by, bw, bh = bm_roi
                    bm_color_roi = color_frame[by:by+bh, bx:bx+bw]
                    try:
                        char_result = reader.readtext(bm_color_roi)
                        char_value, char_text, _ = _parse_ocr_result(char_result)
                        # Extract first A/C/S character
                        valid_chars = {"A", "C", "S"}
                        first_valid = next((c for c in char_text if c.upper() in valid_chars), "")
                        if first_valid:
                            df.loc[df["Frame"] == bm1, "BreathMark_char"] = first_valid.upper()
                    except Exception:
                        pass

            pbar.update(1)

    print(f"[Pass 2] Complete\n")

    return df


# ============================================================
# post_ocr
# ============================================================

def _breath_modes(df):
    """呼吸番号 → 換気モード（"A" / "C" / "S"）。正常フレームが無い呼吸は含まない。"""
    normal = df[(pd.to_numeric(df["Abnormal"], errors="coerce").fillna(0) == 0)
                & (df["BM_Best"].astype(str).str.len() > 0)]
    letters = normal["BM_Best"].astype(str).str[0]
    return letters.groupby(normal["BreathNumber"]).agg(lambda s: s.value_counts().idxmax())


def post_ocr(df, Analyze_Id, user_ID, settings={}):
    """
    Post-process OCR DataFrame and generate breath-level summary.

    Args:
        df: pd.DataFrame from initial_ocr
            Must contain: Frame, VTE, BreathMark, BreathMark_char + _raw/_conf
        Analyze_Id: int
        user_ID: str
        settings: dict from engine_options.json

    Returns:
        (df_frames, df_breaths): tuple of two DataFrames
    """

    df = df.copy()

    df["VTE"] = pd.to_numeric(df["VTE"], errors="coerce")
    if "Ppeak" in df.columns:
        df["Ppeak"] = pd.to_numeric(df["Ppeak"], errors="coerce")
    df["BreathMark"] = pd.to_numeric(df["BreathMark"], errors="coerce").fillna(0).astype(int)

    # Breath detection: cumulative sum of BreathMark transitions
    df["BreathNumber"] = df["BreathMark"].cumsum()
    df.loc[df["BreathNumber"] == 0, "BreathNumber"] = 1

    # Time estimation
    fps = settings.get("post_ocr", {}).get("fps", 10)
    frame_interval = 1 / fps
    df["Time"] = pd.to_timedelta(df["Frame"] * frame_interval, unit="s")

    # Format as HH:MM:SS:CC (CC = centiseconds 0-99)
    def format_time(td):
        total_seconds = int(td.total_seconds())
        centiseconds = int((td.total_seconds() - total_seconds) * 100)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}:{centiseconds:02d}"

    df["Time"] = df["Time"].apply(format_time)

    if "Alert" not in df.columns:
        df["Alert"] = ""

    # Abnormal列はinitial_ocrのPass 1で既にセット済み
    # MeanDiff/StdDiff/HistCorrは互換性のためデフォルト値を追加
    if "Abnormal" not in df.columns:
        df["Abnormal"] = 0
    df["MeanDiff"] = 0.0
    df["StdDiff"] = 0.0
    df["HistCorr"] = 1.0

    df_frames = df

    # Breath-level summary
    df_breaths = df.loc[df.groupby("BreathNumber")["Frame"].idxmin()].copy()

    # 呼吸中に画面に表示されていたVTE（VTE_raw）は、1つ前の呼吸の値（呼気が終わるまでVTEは計算されないため）。
    # そのため呼吸nのVTEは、呼吸n+1の表示値とする。最後の呼吸・次の呼吸の表示値が読めない呼吸は空。
    displayed_vte = (
        df.groupby("BreathNumber")["VTE"]
        .apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
        .sort_index()
    )
    df_breaths["VTE"] = df_breaths["BreathNumber"].map(displayed_vte.shift(-1))

    # 換気モード（A/C/S）: 呼吸中の正常フレームで最も多い BM_Best の文字。
    # 吸気開始の1フレーム目だけ別の文字になることがあるため、BreathMarkのフレーム単独では決めない
    df_breaths["Mode"] = df_breaths["BreathNumber"].map(_breath_modes(df)).fillna("")

    print(f"[post_ocr] {len(df_frames)} frames, {len(df_breaths)} breaths")

    return df_frames, df_breaths


# ============================================================
# extract_waves
# ============================================================

def extract_waves(df_frames, df_breaths, frames_root, output_dir, user_ID,
                  settings={}, overwrite=False, archive_dir=None):
    """
    Extract per-breath waveform CSVs from frame images.
    Breaths containing abnormal frames are skipped.

    Call this after post_ocr():
        df_frames, df_breaths = post_ocr(df, ...)
        effective, df_breaths = extract_waves(df_frames, df_breaths, ...)

    Args:
        df_frames: pd.DataFrame from post_ocr (with Abnormal column)
        df_breaths: pd.DataFrame from post_ocr (1 row per breath)
        frames_root: str path to frame image directory
        output_dir: str path to output CSV directory
        user_ID: str
        settings: dict from engine_options.json
        overwrite: bool
        archive_dir: str path to archive directory (fields.txt書き出し先)

    Returns:
        (effective_breaths_count, df_breaths)
    """
    ew.set_config(settings)
    effective, df_breaths = ew.BreathsToCSVs(
        df_breaths, df_frames, frames_root, output_dir, user_ID,
        overwrite=overwrite,
    )

    # fields.txt をセクション形式で書き出す
    if archive_dir is not None:
        os.makedirs(archive_dir, exist_ok=True)
        fields_path = os.path.join(archive_dir, "fields.txt")
        with open(fields_path, "w", encoding="utf-8") as f:
            f.write("[waves]\n")
            f.write("Pressure\n")
            f.write("Flow\n")
            f.write("Volume\n")
            f.write("\n")
            f.write("[fields]\n")
            f.write("VTE\n")
            f.write("Ppeak\n")
            f.write("Mode\n")
            f.write("BreathMark_char\n")
            f.write("Fragmented\n")
            f.write("Region_X_Start\n")
            f.write("Region_X_End\n")
            f.write("Sweep_Start\n")
            f.write("Sweep_End\n")
            f.write("Alert\n")

    print(f"[extract_waves] {effective} effective breaths (abnormal excluded)")

    return effective, df_breaths
