"""
2フレーム間の差分画像を生成・表示するユーティリティ。

使い方:
  python diff_viewer.py <frames_root> <start_subfolder> <start_frame> <end_subfolder> <end_frame> [--out output.jpg]

例:
  python diff_viewer.py /opt/ventocr/app/temp/admin/Frames 00000 frame_00000149.jpg 00001 frame_00000165.jpg
  python diff_viewer.py /opt/ventocr/app/temp/admin/Frames 00000 frame_00000149.jpg 00000 frame_00000165.jpg --out diff.jpg
"""

import argparse
import os
import cv2
import numpy as np


def generate_diff(start_path, end_path):
    """終了フレームから開始フレームの差分画像を生成する。"""
    img_start = cv2.imread(start_path)
    img_end = cv2.imread(end_path)

    if img_start is None:
        raise FileNotFoundError(f"開始フレームが読み込めません: {start_path}")
    if img_end is None:
        raise FileNotFoundError(f"終了フレームが読み込めません: {end_path}")

    if img_start.shape != img_end.shape:
        raise ValueError(f"フレームサイズが異なります: {img_start.shape} vs {img_end.shape}")

    diff = cv2.absdiff(img_end, img_start)
    return diff, img_start, img_end


def main():
    parser = argparse.ArgumentParser(description="2フレーム間の差分画像を生成")
    parser.add_argument("frames_root", help="フレーム画像のルートディレクトリ")
    parser.add_argument("start_subfolder", help="開始フレームのサブフォルダ名")
    parser.add_argument("start_frame", help="開始フレームのファイル名")
    parser.add_argument("end_subfolder", help="終了フレームのサブフォルダ名")
    parser.add_argument("end_frame", help="終了フレームのファイル名")
    parser.add_argument("--out", default=None, help="出力ファイルパス（省略時は temp に保存）")
    args = parser.parse_args()

    start_path = os.path.join(args.frames_root, args.start_subfolder, args.start_frame)
    end_path = os.path.join(args.frames_root, args.end_subfolder, args.end_frame)

    print(f"開始: {start_path}")
    print(f"終了: {end_path}")

    diff, img_start, img_end = generate_diff(start_path, end_path)

    # 出力先
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diff_output.jpg")

    cv2.imwrite(out_path, diff)
    print(f"差分画像を保存しました: {out_path}")
    print(f"画像サイズ: {diff.shape[1]}x{diff.shape[0]}")


if __name__ == "__main__":
    main()
