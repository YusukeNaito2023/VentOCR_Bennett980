"""
Bennett 980 VentOCR Engine
Reads VTE (tidal volume) and the BreathMark (the breath phase icon at the top left that marks breath boundaries).

Two passes:
- Pass 1: BreathMark detection (every frame, template matching against reference images)
- Pass 2: VTE + BreathMark_char OCR (only frames between BreathMarks)

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
from .detect_sweep import (read_frame, source_scale, reference_shape, crop_reference, detect_sweep_x_band,
                           ROI_X, ROI_W, _BAND_Y0, _BAND_Y1)
import cv2
import re
import pandas as pd
import numpy as np
import os

ENGINE_VERSION = "0.1.6"
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


_READERS = {}  # use_gpu -> Reader. The app's worker processes are reused across segments, so the Reader is not recreated for each segment


def _get_reader(gpu_enabled):
    if gpu_enabled not in _READERS:
        _READERS[gpu_enabled] = _ensure_easyocr_reader_ready(gpu_enabled)
    return _READERS[gpu_enabled]


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


VTE_PAD = 30               # px (4K) of background added around the VTE ROI before OCR (0.1.5)
VTE_MIN_HEIGHT_FRAC = 0.6  # VTE digit boxes lower than this fraction of the tallest box are fragments (e.g. the bottom of the "VTE" label above the value)


def _parse_vte(result):
    """VTE (an integer in mL) from the EasyOCR result: digit boxes at least VTE_MIN_HEIGHT_FRAC of the tallest box, joined left to right without a separator.
    _parse_ocr_result joins two numeric boxes with a decimal point (for decimal values), which turned "417" read as "41" + "7" into 41.7 and a label
    fragment "5" + "379" into 5.379 (0.1.5). A "." inside one box is kept. Returns (value, raw text, confidences of the kept boxes)."""
    raw_texts = [text.replace("_", ".").replace("I", "1") for _, text, _ in result]
    full_text = ", ".join(raw_texts)
    boxes = []
    for (bbox, _, score), text in zip(result, raw_texts):
        digits = re.sub(r"[^0-9.]", "", text)
        if not re.search(r"\d", digits):
            continue
        xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
        boxes.append((min(xs), max(ys) - min(ys), digits, score))
    if not boxes:
        return "", full_text, ""
    tallest = max(b[1] for b in boxes)
    kept = sorted((b for b in boxes if b[1] >= VTE_MIN_HEIGHT_FRAC * tallest), key=lambda b: b[0])
    value = "".join(b[2] for b in kept).strip(".")
    if not re.match(r"^\d+(\.\d+)?$", value):
        value = ""
    return value, full_text, ", ".join(f"{b[3]:.2f}" for b in kept)


def _ocr_vte(reader, color_frame):
    """(VTE, raw text, confidence) read from a color frame scaled to 3840x2160. Empty strings if the ROI is not available."""
    gray_frame = np.dot(color_frame[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    vte_roi = _get_roi_for_frame("VTE", gray_frame.shape)
    if vte_roi is None:
        return "", "", ""
    x, y, w, h = vte_roi
    roi = gray_frame[y:y+h, x:x+w]
    # Pad with the background: the last digit touches the right edge of the ROI, and the OCR detector dropped it ("417" read as "41")
    roi = cv2.copyMakeBorder(roi, VTE_PAD, VTE_PAD, VTE_PAD, VTE_PAD, cv2.BORDER_CONSTANT, value=int(np.median(roi)))
    return _parse_vte(reader.readtext(roi))


def _ocr_ppeak(reader, color_frame):
    """(Ppeak, raw text, confidence) read from a color frame scaled to 3840x2160. Empty strings if the ROI is not available."""
    pp_roi = _get_roi_for_frame("Ppeak", color_frame.shape)
    if pp_roi is None:
        return "", "", ""
    x, y, w, h = pp_roi
    gray = cv2.cvtColor(color_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    # Take only the large number (fragments of the boxed label or the unit can be read as digits)
    res = reader.readtext(gray, allowlist="0123456789")
    res = sorted(res, key=lambda r: -(max(p[1] for p in r[0]) - min(p[1] for p in r[0])))[:1]
    return _parse_ocr_result(res)


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
    Two-pass OCR:
    - Pass 1: BreathMark transition detection (every frame, received one by one from the color frame generator)
    - Pass 2: VTE + BreathMark_char OCR (frames between BreathMarks, read from disk in frames_dir)

    Args:
        frames: generator of color frames (BGR, numpy arrays). Do not convert to a list.
        total_frames: int
        subfolder: str
        settings: dict
        progress_callback: callable or None
        frames_dir: str folder of saved JPEG frames (for random access)

    Returns:
        pd.DataFrame with columns: Frame, VTE, VTE_raw, VTE_conf, BreathMark, BreathMark_raw, BreathMark_char
    """

    ocr_settings = settings.get("initial_ocr", {})
    gpu_enabled = ocr_settings.get("use_gpu", True)

    reader = _get_reader(gpu_enabled)
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
    # Color histograms stop matching the references when the capture device shifts the colors (e.g. the 1280x720 videos from 2026-09),
    # so the shape of the A/C/S Start/End references is compared with normalized cross-correlation, which is robust to brightness and contrast.
    # TM_CCOEFF_NORMED between images of the same size is the cosine between mean-subtracted vectors, so the 6 references are kept as a mean-subtracted, normalized matrix
    # and one matrix product per frame gives the 6 scores (1.3.3; about 30x faster than calling cv2.matchTemplate 6 times; differences around 1e-6)
    _ref_grays = {}  # {"A_Start": gray image, "A_End": gray image, ...}
    _ref_names, _ref_mat = [], None
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
                "Sweep_X": np.nan,  # x of the pen tip (sweep line) in original video pixels. Used for the sweep line in the app's capture view. Empty if not detected
                # BM_BestScore clipped to 0-1. Used by the app's check of the first 50 frames (manifest frame_check_columns).
                # Values below 0.5 count as "not read" (0.75 or higher on the PB980 screen, 0.11 or lower on other ventilators' screens)
                "BM_Best_conf": np.nan,
            }

            if color_frame is not None:
                # Frames such as 1280x720 are upscaled to 3840x2160 and read in 4K coordinates. Only the sweep band and the BreathMark region are upscaled, not the whole frame
                scale = source_scale(color_frame)
                ref_shape = reference_shape(color_frame)
                sweep_x = 0
                if ref_shape[0] >= _BAND_Y1 and ref_shape[1] >= ROI_X + ROI_W:
                    sweep_x = detect_sweep_x_band(crop_reference(color_frame, ROI_X, _BAND_Y0, ROI_W, _BAND_Y1 - _BAND_Y0))
                if sweep_x > 0:
                    row["Sweep_X"] = round(sweep_x * scale, 1)

                if frame_shape is None:
                    frame_shape = ref_shape
                    detected_resolution = (frame_shape[0], frame_shape[1])
                    src_h, src_w = color_frame.shape[:2]
                    if (src_h, src_w) == detected_resolution:
                        print(f"[info] Detected resolution: {src_w}x{src_h}")
                    else:
                        print(f"[info] Detected resolution: {src_w}x{src_h} (read upscaled to {detected_resolution[1]}x{detected_resolution[0]})")

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
                    if _ref_grays:
                        _ref_names = list(_ref_grays)
                        _ref_mat = np.stack([_ref_grays[n].astype(np.float64).ravel() for n in _ref_names])
                        _ref_mat -= _ref_mat.mean(axis=1, keepdims=True)
                        _ref_mat /= np.linalg.norm(_ref_mat, axis=1, keepdims=True)
                    print(f"[BreathMark] Loaded {len(_ref_grays)} reference images: {list(_ref_grays.keys())}")

                bm_roi = _get_roi_for_frame("BreathMark", ref_shape)
                if bm_roi is not None:
                    bm_color_roi = crop_reference(color_frame, *bm_roi)

                    # Every frame: template matching against all 6 A/C/S Start/End references (normalized cross-correlation)
                    if _ref_grays and bm_color_roi.shape[:2] == next(iter(_ref_grays.values())).shape:
                        g = cv2.cvtColor(bm_color_roi, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
                        g -= g.mean()
                        norm = np.linalg.norm(g)
                        scores = _ref_mat @ g / norm if norm > 0 else np.zeros(len(_ref_names))
                        best_name = ""
                        best_score = -1.0
                        for name, score in zip(_ref_names, scores):
                            score = float(score)
                            row[f"BM_{name}"] = round(score, 4)
                            if score > best_score:
                                best_score = score
                                best_name = name
                        row["BM_Best"] = best_name
                        row["BM_BestScore"] = round(best_score, 4)
                        row["BM_Best_conf"] = round(min(max(best_score, 0.0), 1.0), 4)
                        if best_score < Settings.BREATHMARK_ABNORMAL_SCORE:
                            row["Abnormal"] = 1

                        # BreathMark detection: an _End -> _Start transition marks the start of inspiration
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

            # OCR VTE: read the color JPEG from frames_dir, convert to grayscale and run OCR (_ocr_vte; 0.1.5 parses it as an integer)
            mid_color_frame = _load_color_frame(frames_dir, midpoint) if frames_dir is not None else None
            if mid_color_frame is not None:
                try:
                    vte_value, vte_raw, vte_conf = _ocr_vte(reader, mid_color_frame)
                    df.loc[df["Frame"] == bm1, "VTE"] = vte_value
                    df.loc[df["Frame"] == bm1, "VTE_raw"] = vte_raw
                    df.loc[df["Frame"] == bm1, "VTE_conf"] = vte_conf
                except Exception:
                    pass

            # OCR Ppeak: the value shown at the end of the breath (the frame of the next BreathMark) belongs to this breath (it updates when inspiration ends)
            end_color_frame = _load_color_frame(frames_dir, bm2) if frames_dir is not None else None
            if end_color_frame is not None:
                try:
                    pp_value, pp_raw, pp_conf = _ocr_ppeak(reader, end_color_frame)
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
    """Breath number -> ventilation mode ("A" / "C" / "S"). Breaths without normal frames are not included."""
    normal = df[(pd.to_numeric(df["Abnormal"], errors="coerce").fillna(0) == 0)
                & (df["BM_Best"].astype(str).str.len() > 0)]
    letters = normal["BM_Best"].astype(str).str[0]
    return letters.groupby(normal["BreathNumber"]).agg(lambda s: s.value_counts().idxmax())


def _vte_of_breaths(df_frames):
    """Breath number -> VTE: the value shown during the next breath (the last value read in that breath; VTE is shown after expiration ends)."""
    displayed = (
        pd.to_numeric(df_frames["VTE"], errors="coerce").groupby(df_frames["BreathNumber"])
        .apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
        .sort_index()
    )
    return displayed.shift(-1)


NO_SCREEN_ABNORMAL_RATIO = 0.9  # With no breath boundaries and at least this fraction of Abnormal frames, the video is not a PB980 screen (post_ocr)


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

    # Videos without the PB980 screen (other ventilators, etc.) match no reference in any frame (Abnormal), so no breath boundary is found.
    # Continuing would "complete" the whole video as one breath, so the analysis fails (judged on the whole video, not per segment, so segments showing only the settings screen do not stop it)
    abnormal = pd.to_numeric(df["Abnormal"], errors="coerce").fillna(0) if "Abnormal" in df.columns else pd.Series(0, index=df.index)
    if len(df) and df["BreathMark"].sum() == 0 and abnormal.mean() >= NO_SCREEN_ABNORMAL_RATIO:
        raise RuntimeError(f"PB980 screen not found (no breath boundaries; in {abnormal.mean() * 100:.0f}% of the frames the icon at the top left "
                           "matches no reference). Check that the video shows the PB980 screen; for another ventilator, choose that ventilator's engine")

    # 0.1.5: BreathMarks at segment boundaries. Pass 1 runs per segment and does not carry the last icon of the previous segment over,
    # so a breath whose _End -> _Start transition falls on the first frame of a segment was missed (which breaths depended on where the segments were cut).
    # Look at the whole sequence of BM_Best: where the segment changes between two frames, add the mark that Pass 1 could not see
    df = df.sort_values("Frame", kind="stable").reset_index(drop=True)
    df["BoundaryMark"] = 0
    if "BM_Best" in df.columns and "source_file" in df.columns and len(df) > 1:
        best = df["BM_Best"].astype(str)
        prev = best.shift(1)
        boundary = df["source_file"].ne(df["source_file"].shift(1))
        boundary.iloc[0] = False
        added = boundary & prev.str.contains("_End", na=False) & best.str.contains("_Start", na=False) & (df["BreathMark"] == 0)
        df.loc[added, "BreathMark"] = 1
        df.loc[added, "BoundaryMark"] = 1
        df.loc[added, "BreathMark_raw"] = prev[added] + " -> " + best[added] + " (segment boundary)"
        if added.any():
            print(f"[post_ocr] {int(added.sum())} BreathMarks added at segment boundaries")

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

    # The Abnormal column is already set in Pass 1 of initial_ocr
    # MeanDiff/StdDiff/HistCorr get default values for compatibility
    if "Abnormal" not in df.columns:
        df["Abnormal"] = 0
    df["MeanDiff"] = 0.0
    df["StdDiff"] = 0.0
    df["HistCorr"] = 1.0

    df_frames = df

    # Breath-level summary
    df_breaths = df.loc[df.groupby("BreathNumber")["Frame"].idxmin()].copy()

    # The VTE shown on screen during a breath (VTE_raw) is the previous breath's value (VTE is not computed until expiration ends).
    # So the VTE of breath n is the value shown during breath n+1. Empty for the last breath and when the next breath's value cannot be read.
    df_breaths["VTE"] = df_breaths["BreathNumber"].map(_vte_of_breaths(df))

    # Ventilation mode (A/C/S): the most frequent BM_Best letter among the breath's normal frames.
    # The first frame of inspiration can show a different letter, so the BreathMark frame alone does not decide it
    df_breaths["Mode"] = df_breaths["BreathNumber"].map(_breath_modes(df)).fillna("")

    print(f"[post_ocr] {len(df_frames)} frames, {len(df_breaths)} breaths")

    return df_frames, df_breaths


# ============================================================
# extract_waves
# ============================================================

def _fill_unread_values(df_frames, df_breaths, frames_root, settings):
    """0.1.5: read VTE and Ppeak for breaths that Pass 2 could not read, from the whole-video frames (frames_root = Frames/_flat).
    Pass 2 runs per segment, so a breath whose start and next start are in different segments (it crosses a segment boundary) and a breath
    added at a boundary by post_ocr were never read; their VTE (and the VTE of the previous breath, which is shown during this breath) was empty,
    and which breaths were empty depended on where the segments were cut. Only those breaths are read (about 2 frames each), so the cost is small.
    df_frames and df_breaths are updated in place (the app writes Frames.csv after extract_waves). Returns (frames read, seconds)."""
    import time

    t0 = time.perf_counter()
    if "source_file" not in df_frames.columns or len(df_breaths) < 2:
        return 0, 0.0
    row_of = dict(zip(df_frames["Frame"], df_frames.index))
    seg_of = dict(zip(df_frames["Frame"], df_frames["source_file"]))
    starts = [int(f) for f in sorted(df_breaths["Frame"])]
    targets = []
    for f1, f2 in zip(starts[:-1], starts[1:]):
        r1 = row_of.get(f1)
        if r1 is None:
            continue
        added = "BoundaryMark" in df_frames.columns and df_frames.at[r1, "BoundaryMark"] == 1
        if seg_of.get(f1) == seg_of.get(f2) and not added:
            continue
        need_vte = pd.isna(pd.to_numeric(df_frames.at[r1, "VTE"], errors="coerce"))
        need_pp = "Ppeak" in df_frames.columns and pd.isna(pd.to_numeric(df_frames.at[r1, "Ppeak"], errors="coerce"))
        if need_vte or need_pp:
            targets.append((f1, f2, need_vte, need_pp))
    if not targets:
        return 0, 0.0
    for frame in (df_frames, df_breaths):  # text columns (empty ones are read back as float)
        for col in ("VTE_raw", "VTE_conf", "Ppeak_raw", "Ppeak_conf"):
            if col in frame.columns and frame[col].dtype != object:
                frame[col] = frame[col].astype(object)
    reader = _get_reader(settings.get("initial_ocr", {}).get("use_gpu", True))
    t_reader = time.perf_counter() - t0
    n_frames = 0
    for f1, f2, need_vte, need_pp in targets:
        r1 = row_of[f1]
        cols = {}
        if need_vte:
            img = read_frame(os.path.join(frames_root, f"frame_{(f1 + f2) // 2:08d}.jpg"))
            n_frames += 1
            if img is not None:
                try:
                    value, raw, conf = _ocr_vte(reader, img)
                    cols.update(VTE=pd.to_numeric(value, errors="coerce"), VTE_raw=raw, VTE_conf=conf)
                except Exception:
                    pass
        if need_pp:
            img = read_frame(os.path.join(frames_root, f"frame_{f2:08d}.jpg"))
            n_frames += 1
            if img is not None:
                try:
                    value, raw, conf = _ocr_ppeak(reader, img)
                    cols.update(Ppeak=pd.to_numeric(value, errors="coerce"), Ppeak_raw=raw, Ppeak_conf=conf)
                except Exception:
                    pass
        for col, value in cols.items():
            df_frames.at[r1, col] = value
            if col != "VTE":  # the breath's VTE is recomputed below (it is the value shown during the next breath)
                df_breaths.loc[df_breaths["Frame"] == f1, col] = value
    df_breaths["VTE"] = df_breaths["BreathNumber"].map(_vte_of_breaths(df_frames))
    elapsed = time.perf_counter() - t0
    print(f"[extract_waves] read {n_frames} frames for {len(targets)} breaths that Pass 2 could not read "
          f"({elapsed:.1f} s, of which loading the OCR model {t_reader:.1f} s)")
    return n_frames, elapsed

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
        archive_dir: str path to archive directory (where fields.txt is written)

    Returns:
        (effective_breaths_count, df_breaths)
    """
    ew.set_config(settings)
    _fill_unread_values(df_frames, df_breaths, frames_root, settings)
    effective, df_breaths = ew.BreathsToCSVs(
        df_breaths, df_frames, frames_root, output_dir, user_ID,
        overwrite=overwrite,
    )

    # Write fields.txt in sections
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
