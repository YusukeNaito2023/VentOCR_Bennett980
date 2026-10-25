"""
Utility to create and show the difference image between two frames.

Usage:
  python diff_viewer.py <frames_root> <start_subfolder> <start_frame> <end_subfolder> <end_frame> [--out output.jpg]

Examples:
  python diff_viewer.py /opt/ventocr/app/temp/admin/Frames 00000 frame_00000149.jpg 00001 frame_00000165.jpg
  python diff_viewer.py /opt/ventocr/app/temp/admin/Frames 00000 frame_00000149.jpg 00000 frame_00000165.jpg --out diff.jpg
"""

import argparse
import os
import cv2
import numpy as np


def generate_diff(start_path, end_path):
    """Creates the difference image of the end frame minus the start frame."""
    img_start = cv2.imread(start_path)
    img_end = cv2.imread(end_path)

    if img_start is None:
        raise FileNotFoundError(f"Cannot read the start frame: {start_path}")
    if img_end is None:
        raise FileNotFoundError(f"Cannot read the end frame: {end_path}")

    if img_start.shape != img_end.shape:
        raise ValueError(f"Frame sizes differ: {img_start.shape} vs {img_end.shape}")

    diff = cv2.absdiff(img_end, img_start)
    return diff, img_start, img_end


def main():
    parser = argparse.ArgumentParser(description="Create the difference image between two frames")
    parser.add_argument("frames_root", help="Root folder of the frame images")
    parser.add_argument("start_subfolder", help="Subfolder of the start frame")
    parser.add_argument("start_frame", help="File name of the start frame")
    parser.add_argument("end_subfolder", help="Subfolder of the end frame")
    parser.add_argument("end_frame", help="File name of the end frame")
    parser.add_argument("--out", default=None, help="Output file path (saved in temp if omitted)")
    args = parser.parse_args()

    start_path = os.path.join(args.frames_root, args.start_subfolder, args.start_frame)
    end_path = os.path.join(args.frames_root, args.end_subfolder, args.end_frame)

    print(f"Start: {start_path}")
    print(f"End: {end_path}")

    diff, img_start, img_end = generate_diff(start_path, end_path)

    # Output path
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diff_output.jpg")

    cv2.imwrite(out_path, diff)
    print(f"Saved the difference image: {out_path}")
    print(f"Image size: {diff.shape[1]}x{diff.shape[0]}")


if __name__ == "__main__":
    main()
