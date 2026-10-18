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
    if (w, h) == (REF_W, REF_H) or abs(w * REF_H - h * REF_W) > 0.01 * w * REF_H:
        return img
    return cv2.resize(img, (REF_W, REF_H), interpolation=cv2.INTER_CUBIC)


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
    img = read_frame(frame_path)
    if img is None:
        return 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 全ROIで「グリッド除外V_max <= 閾値」の列を計算
    low_vmax = np.ones(ROI_W, dtype=bool)
    for name, ry, rh in ROIS:
        roi = hsv[ry:ry+rh, ROI_X:ROI_X+ROI_W]

        # 異常フレーム判定: 各ROIで波形が最低100列必要
        wave = (
            (roi[:, :, 0] >= WAVE_H_MIN) & (roi[:, :, 0] <= WAVE_H_MAX) &
            (roi[:, :, 1] >= WAVE_S_MIN) &
            (roi[:, :, 2] >= WAVE_V_MIN)
        )
        if (wave.sum(axis=0) >= 1).sum() < MIN_WAVE_COLS_PER_ROI:
            return 0

        # グリッド線を除外してV_maxを計算
        grid_mask = (roi[:, :, 0] >= GRID_H_MIN) & (roi[:, :, 0] <= GRID_H_MAX)
        v_ch = roi[:, :, 2].copy()
        v_ch[grid_mask] = 0
        col_vmax = v_ch.max(axis=0)
        low_vmax &= (col_vmax <= VMAX_THRESH)

    # 連続する低V_max領域をグループ化
    groups = []
    in_gap = False
    gs = 0
    for i in range(ROI_W):
        if low_vmax[i] and not in_gap:
            gs = i
            in_gap = True
        elif (not low_vmax[i]) and in_gap:
            groups.append((gs, i - 1))
            in_gap = False
    if in_gap:
        groups.append((gs, ROI_W - 1))

    # フィルタ: 幅>=MIN_SWEEP_WIDTH、左側にV_max>閾値の列がある
    valid = []
    for gs, ge in groups:
        width = ge - gs + 1
        if width < MIN_SWEEP_WIDTH:
            continue
        if gs < 5:
            continue
        left_ok = not low_vmax[max(0, gs - 5):gs].all()
        if not left_ok:
            continue
        valid.append((gs, ge, width))

    if not valid:
        return 0

    # 複数候補: 左側のV_maxが最も高いもの（アクティブ波形の直後）を選択
    if len(valid) == 1:
        gs, ge, width = valid[0]
    else:
        best = None
        best_left_v = 0
        for gs, ge, width in valid:
            left_region = []
            for name, ry, rh in ROIS:
                roi = hsv[ry:ry+rh, ROI_X:ROI_X+ROI_W]
                grid_mask = (roi[:, :, 0] >= GRID_H_MIN) & (roi[:, :, 0] <= GRID_H_MAX)
                v_ch = roi[:, :, 2].copy()
                v_ch[grid_mask] = 0
                left_vmax = v_ch[:, max(0, gs - 20):gs].max()
                left_region.append(left_vmax)
            avg_left_v = np.mean(left_region)
            if avg_left_v > best_left_v:
                best_left_v = avg_left_v
                best = (gs, ge, width)
        gs, ge, width = best

    sweep_start = gs + ROI_X
    return sweep_start


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
    img = read_frame(frame_path)
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
