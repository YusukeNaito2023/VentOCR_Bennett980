"""Build the axis-label templates (axis_labels.npz) for reading the waveform scales (axis_labels.py, 0.1.6).

1. python tools/build_axis_labels.py extract SPEC.json STRIPS.npz
   Takes frames from the videos in SPEC.json (ffmpeg, read-only) every `every_sec` seconds and keeps the label column
   as the engine sees it: JPEG quality 95 (as the app's cv2.imwrite), scaled to the 3840x2160 reference (read_frame),
   then to BASE (axis_labels.base_strip).
2. python tools/build_axis_labels.py build STRIPS.npz OUT.npz
   For every group and label slot, the boxes around the label (centred at the text centroid) are averaged into one
   template, leaving out boxes that differ from the group median by more than CLUSTER_RMS (a settings menu or a pop-up
   over the labels, a waveform crossing the label). Writes OUT.npz and OUT.png (all templates, for checking by eye).
3. python tools/build_axis_labels.py check STRIPS.npz OUT.npz
   Reads every frame with the templates: what is read per band, and the RMS of the best template of the same text and
   of another text (to set MAX_RMS and MIN_GAP).

SPEC.json: {"groups": [{"name": "...", "videos": ["..."], "start": 10, "every_sec": 60,
                        "labels": {"Pressure_top": "40", "Pressure_zero": "0", ...}}, ...]}
The labels of a group must be the same in all its frames (look at the frames first; a frame whose labels differ is
left out by CLUSTER_RMS and reported).
Run it from anywhere; it imports axis_labels.py and detect_sweep.py from the engine directory.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import axis_labels as al  # noqa: E402
from detect_sweep import to_reference_size  # noqa: E402

CLUSTER_RMS = 6.0


def _probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                          "-of", "json", video], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return j["streams"][0]["width"], j["streams"][0]["height"], float(j["format"]["duration"])


def _frame(video, t, w, h):
    buf = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    if len(buf) < w * h * 3:
        return None
    img = np.frombuffer(buf[:w * h * 3], np.uint8).reshape(h, w, 3)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1], cv2.IMREAD_COLOR)
    return to_reference_size(img)


def extract(spec_path, out):
    spec = json.load(open(spec_path))
    strips, names, group_of = [], [], []
    for gi, g in enumerate(spec["groups"]):
        for video in g["videos"]:
            w, h, dur = _probe(video)
            t = float(g.get("start", 10))
            while t < dur:
                ref = _frame(video, t, w, h)
                if ref is not None:
                    strips.append(al.base_strip(ref))
                    names.append(f"{os.path.basename(video)}@{t:.0f}")
                    group_of.append(gi)
                t += float(g.get("every_sec", 60))
            print(f"{g['name']}: {os.path.basename(video)} done ({len(strips)} frames so far)")
    np.savez_compressed(out, strips=np.array(strips), names=np.array(names), group_of=np.array(group_of),
                        spec=json.dumps(spec))
    print(f"{len(strips)} frames -> {out}")


def _load(path):
    z = np.load(path)
    return z["strips"], [str(n) for n in z["names"]], z["group_of"], json.loads(str(z["spec"]))


def build(strips_path, out):
    strips, names, group_of, spec = _load(strips_path)
    texts = {b: [] for b in al.BANDS}
    temps = {b: [] for b in al.BANDS}
    sources = {b: [] for b in al.BANDS}
    for gi, g in enumerate(spec["groups"]):
        for slot, text in g["labels"].items():
            band = slot.split("_")[0]
            boxes = []
            for s in strips[group_of == gi]:
                f = al.find_label(s, slot)
                if f is not None and not f[1].any():
                    boxes.append(al.normalized(f[0]))
            if not boxes:
                print(f"{g['name']} {slot}: no clean box")
                continue
            boxes = np.array(boxes)
            med = np.median(boxes, axis=0)
            rms = np.sqrt(((boxes - med) ** 2).mean(axis=(1, 2)))
            keep = rms <= CLUSTER_RMS
            print(f"{g['name']} {slot} '{text}': {int(keep.sum())}/{len(boxes)} boxes kept (RMS to median: max kept {rms[keep].max():.2f})")
            texts[band].append(text)
            temps[band].append(boxes[keep].mean(axis=0))
            sources[band].append(g["name"])
    np.savez_compressed(out, **{f"texts_{b}": np.array(texts[b]) for b in al.BANDS if texts[b]},
                        **{f"templates_{b}": np.array(temps[b], np.float32) for b in al.BANDS if temps[b]},
                        **{f"sources_{b}": np.array(sources[b]) for b in al.BANDS if sources[b]})
    # One picture of all templates (scaled x4, background 30)
    tiles = []
    for b in al.BANDS:
        for t, tp, src in zip(texts[b], temps[b], sources[b]):
            im = cv2.resize(np.clip(tp + 30, 0, 255).astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            lab = np.zeros((18, im.shape[1]), np.uint8)
            cv2.putText(lab, f"{b[0]} {t} {src[:6]}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
            tiles.append(np.vstack([lab, im]))
    cv2.imwrite(os.path.splitext(out)[0] + ".png", np.hstack([np.pad(t, ((0, 0), (0, 4))) for t in tiles]))
    print(f"-> {out}")


def check(strips_path, npz):
    strips, names, group_of, spec = _load(strips_path)
    reader = al.AxisLabelReader(npz)
    same, other = [], []
    read_count = {b: 0 for b in al.BANDS}
    wrong = []
    geo = {}
    for s, n, gi in zip(strips, names, group_of):
        want = spec["groups"][gi]["labels"]
        # RMS of the best template of the right text and of any other text, per slot (on the found box)
        for slot, text in want.items():
            band = slot.split("_")[0]
            f = al.find_label(s, slot)
            if f is None or f[1].mean() > al.MAX_MASKED:
                continue
            ts, tp = reader.sets[band]
            r = np.sqrt(((al.normalized(f[0], f[1]) - tp)[:, ~f[1]] ** 2).mean(axis=1))
            mine = [x for x, t in zip(r, ts) if t == text]
            rest = [x for x, t in zip(r, ts) if t != text]
            if mine:
                same.append(min(mine))
            if rest:
                other.append(min(rest))
        # What the engine reads
        got = reader.read_strip(s, 2160 / al.BASE_H)
        for b, v in got.items():
            read_count[b] += 1
            geo.setdefault((spec["groups"][gi]["name"], b), []).append((v["k"], v["zero_y"], v["top_y"], v["bottom_y"]))
            for slot, val in v["labels"].items():
                if slot in want and val != al.text_value(want[slot], slot):
                    wrong.append((n, slot, val, want[slot]))
    n = len(strips)
    print(f"frames {n}; bands read: " + ", ".join(f"{b} {c}/{n}" for b, c in read_count.items()))
    if same:
        print(f"RMS, same text: median {np.median(same):.2f}, 99% {np.percentile(same, 99):.2f}, max {max(same):.2f}")
    if other:
        print(f"RMS, another text: min {min(other):.2f}, 1% {np.percentile(other, 1):.2f}")
    print(f"labels read wrong: {len(wrong)}", wrong[:10])
    print("scale per group and band (reference px): units/px, zero row, top label row, bottom label row (median, max - min)")
    for (g, b), v in sorted(geo.items()):
        v = np.array(v)
        print(f"  {g} {b}: " + ", ".join(f"{np.median(v[:, j]):.4f} ({np.ptp(v[:, j]):.4f})" for j in range(4)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "extract" and len(sys.argv) == 4:
        extract(sys.argv[2], sys.argv[3])
    elif cmd == "build" and len(sys.argv) == 4:
        build(sys.argv[2], sys.argv[3])
    elif cmd == "check" and len(sys.argv) == 4:
        check(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
