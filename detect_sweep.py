"""
Bennett 980 捜査線（sweep line）検出ユーティリティ。

JPEGフレーム画像を指定すると、捜査線のX軸位置を返す。
検出できない場合は0を返す。

捜査線の検出原理:
  全3ROI(Pressure/Flow/Volume)で、グリッド線(H=90-130)を除外した
  V_maxが閾値以下になる列が MIN_SWEEP_WIDTH px以上連続する帯 = 捜査線。
  幅が10pxを超える場合は左端から10pxの範囲を返す。

使い方:
  python detect_sweep.py <frame_path>
  python detect_sweep.py /opt/ventocr/app/temp/admin/Frames/00001/frame_00000461.jpg
"""

import sys
import cv2
import numpy as np


# 3840x2160用ROI定義（X開始=520, 幅=2877で統一）
ROIS = [
    ("Pressure", 269, 324),  # (y, h)
    ("Flow",     733, 324),
    ("Volume",  1177, 324),
]
ROI_X = 520
ROI_W = 2877

# 捜査線の帯の最小幅（px）。4K の取り込みでは 11〜14px、1280x720 を 4K に拡大したものでは 8〜11px
MIN_SWEEP_WIDTH = 7

# 捜査線内のV_max上限（グリッド線除外後）
VMAX_THRESH = 45

# グリッド線のH範囲（V_max計算から除外する）
GRID_H_MIN = 90
GRID_H_MAX = 130

# 各ROIで最低限の波形列数（異常フレーム判定）
MIN_WAVE_COLS_PER_ROI = 100

# 波形ピクセルのHSV条件
WAVE_H_MIN = 55
WAVE_H_MAX = 75
WAVE_S_MIN = 80
WAVE_V_MIN = 30

# エンジンの座標・閾値はすべて 3840x2160 のフレームで決めてある
REF_W, REF_H = 3840, 2160


def to_reference_size(img):
    """3840x2160 以外の 16:9 のフレーム（1280x720・1920x1080 など）を 3840x2160 に拡大する。
    PB980 の画面をそのまま取り込んだ動画は解像度が違っても配置が同じなので、拡大すれば 4K の座標・閾値がそのまま使える。
    16:9 でないもの（読めなかったものを含む）はそのまま返す。"""
    if img is None:
        return None
    h, w = img.shape[:2]
    if to_reference_size_factor(w, h) == 1.0:
        return img
    return cv2.resize(img, (REF_W, REF_H), interpolation=cv2.INTER_CUBIC)


def source_scale(img):
    """元のフレームの画素 / 3840x2160 にそろえたフレームの画素（to_reference_size で拡大しないものは 1）。
    エンジンが書き出す画素の値（Sweep_X・Sweep_Start など）は、これを掛けて元の動画の画素に戻す（アプリは元の動画の幅で位置を出す）。"""
    if img is None:
        return 1.0
    return to_reference_size_factor(img.shape[1], img.shape[0])


def to_reference_size_factor(w, h):
    """幅 w・高さ h のフレームを to_reference_size で 3840x2160 にするときの、元 / 拡大後 の比（拡大しないものは 1）。"""
    if (w, h) == (REF_W, REF_H) or abs(w * REF_H - h * REF_W) > 0.01 * w * REF_H:
        return 1.0
    return w / REF_W


def reference_shape(img):
    """to_reference_size したときのフレームの形（拡大するものは 3840x2160、しないものはそのまま）。"""
    h, w = img.shape[:2]
    return (REF_H, REF_W) + img.shape[2:] if to_reference_size_factor(w, h) != 1.0 else img.shape


def crop_reference(img, x, y, w, h, margin=4):
    """to_reference_size したフレームの (x, y, w, h) の領域。全画面を拡大せず、その領域だけを元のフレームから切り出して拡大する
    （Pass 1 は毎フレーム読むので、1280x720 の全画面を 4K に拡大する時間を省く）。
    拡大の倍率が整数（1280x720 は 3、1920x1080 は 2）なら、余白を付けて切り出せば位置がそろい、全画面を拡大して切り出したものと
    ほぼ同じ画素になる（差は一部の画素で最大 1 階調）。整数でない倍率は全画面を拡大して切り出す。"""
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
    """フレーム画像を読み、3840x2160 にそろえる（to_reference_size）。読めなければ None。"""
    return to_reference_size(cv2.imread(frame_path))


def detect_sweep_x(frame_path):
    """
    フレーム画像から捜査線のX位置を検出する。

    Args:
        frame_path: str, JPEGフレーム画像のパス

    Returns:
        int: 捜査線の開始X座標（絶対座標）。検出不可時は0。
    """
    return detect_sweep_x_img(read_frame(frame_path))


# 3つの波形ROIを含む帯（HSV への変換をこの範囲だけにする）
_BAND_Y0 = min(ry for _, ry, _ in ROIS)
_BAND_Y1 = max(ry + rh for _, ry, rh in ROIS)


def detect_sweep_x_img(img):
    """3840x2160 にそろえたフレーム（BGR）から捜査線のX位置を検出する（detect_sweep_x と同じ。検出不可時は0）。"""
    if img is None or img.shape[0] < _BAND_Y1 or img.shape[1] < ROI_X + ROI_W:
        return 0
    return detect_sweep_x_band(img[_BAND_Y0:_BAND_Y1, ROI_X:ROI_X + ROI_W])


def detect_sweep_x_band(band):
    """3つの波形ROIを含む帯（4K の座標で x = ROI_X〜ROI_X+ROI_W、y = _BAND_Y0〜_BAND_Y1。BGR）から捜査線のX位置を検出する。
    HSV への変換は ROI の行だけ、列のまとまりは numpy で探す（1.3.3。結果は以前の detect_sweep_x_img と同じ）。"""
    # 全ROIで「グリッド除外V_max <= 閾値」の列を計算
    low_vmax = np.ones(ROI_W, dtype=bool)
    v_rois = []
    for name, ry, rh in ROIS:
        roi = cv2.cvtColor(band[ry - _BAND_Y0:ry - _BAND_Y0 + rh], cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]

        # 異常フレーム判定: 各ROIで波形が最低100列必要
        wave = (h_ch >= WAVE_H_MIN) & (h_ch <= WAVE_H_MAX) & (s_ch >= WAVE_S_MIN) & (v_ch >= WAVE_V_MIN)
        if np.count_nonzero(wave.any(axis=0)) < MIN_WAVE_COLS_PER_ROI:
            return 0

        # グリッド線を除外してV_maxを計算
        v_grid = np.where((h_ch >= GRID_H_MIN) & (h_ch <= GRID_H_MAX), 0, v_ch)
        v_rois.append(v_grid)
        low_vmax &= v_grid.max(axis=0) <= VMAX_THRESH

    # 連続する低V_max領域（gs〜ge）を探し、幅>=MIN_SWEEP_WIDTH、左側にV_max>閾値の列があるものだけ残す
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

    # 複数候補: 左側のV_maxが最も高いもの（アクティブ波形の直後）を選択
    if len(valid) == 1:
        gs = valid[0]
    else:
        gs, best_left_v = None, 0
        for cand in valid:
            avg_left_v = np.mean([v[:, max(0, cand - 20):cand].max() for v in v_rois])
            if avg_left_v > best_left_v:
                best_left_v, gs = avg_left_v, cand

    return gs + ROI_X


# 設定変更画面の判定用定数
_SETTINGS_CHECK_ROI_4K = (520, 269, 2877, 324)     # Pressure ROI (3840x2160)
_SETTINGS_CHECK_ROI_FHD = (260, 130, 1474, 186)    # Pressure ROI (1920x1080)
_SETTINGS_BRIGHT_V_THRESH = 50
_SETTINGS_BRIGHT_RATIO_THRESH = 0.3


def is_settings_screen(frame_path):
    """
    設定変更画面かどうかを判定する。
    波形ROI内のV値（明度）分布から、ダイアログ表示による
    異常な明るさを検出する。

    Args:
        frame_path: str, JPEGフレーム画像のパス

    Returns:
        bool: 設定変更画面ならTrue
    """
    return is_settings_screen_img(read_frame(frame_path))


def is_settings_screen_img(img):
    """is_settings_screen と同じ判定を、読み込んだフレーム（3840x2160 にそろえたもの）で行う。"""
    if img is None:
        return False

    h, w = img.shape[:2]
    rx, ry, rw, rh = _SETTINGS_CHECK_ROI_4K if w == 3840 else _SETTINGS_CHECK_ROI_FHD
    roi_bgr = img[ry:ry+rh, rx:rx+rw]
    # V = max(B, G, R) — HSV変換なしでV値を取得
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
