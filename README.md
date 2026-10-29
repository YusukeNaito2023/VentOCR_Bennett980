# Bennett 980 VentOCR Engine

Analyzes Paw (Airway Pressure), Flow, and Volume from Bennett 980 ventilator display.

## Setup

1. Capture a screenshot of the Bennett 980 display
2. Place it as `sample_frame.jpg` in this directory
3. Open the image and measure pixel coordinates for each ROI
4. Update `Settings.py` with the correct coordinates

## ROI Configuration (Settings.py)

```python
ROIs = {
    "Paw":    [x, y, width, height],
    "Flow":   [x, y, width, height],
    "Volume": [x, y, width, height],
}
```

Coordinates are measured from the top-left corner of the display image.

## Waveform scales (0.1.6)

The scale and the zero of each waveform are read from the axis labels at the left of each band, in the end frame of every breath
(axis_labels.py, templates in axis_labels.npz). Each label (top, "0", bottom) is compared as a whole with the labels seen before and
located to a fraction of a pixel; value per pixel comes from the labels and the zero from the zero baseline (or the label "0").
The crop of each band covers its labels, so expiratory Flow and negative Paw / Volume are no longer cut off.
When the labels of a band are not read (a label not seen before, a menu over the labels), its scale is estimated as before 0.1.6
(from VTE / PPEAK and the neighbouring breaths) and the breath gets the Alert "Axis". The templates are built with
tools/build_axis_labels.py (2026-10-29: Paw 40 / 0 / -20, Flow 80 and 120, Volume 1000 / 0 / -100, from 720p captures and 980_sample).

## Alert (the Alert column of Breaths.csv)

| Value | Meaning | Accuracy evaluation |
|---|---|---|
| Fragment | Cut at the right edge of the screen, or the first breath of the recording (partial waveform) | Excluded |
| Abnormal | Contains frames whose breath icon (top left) matches no reference (settings screen, disconnection, etc.) | Excluded |
| No Sweep | Sweep line not found; waveform not extracted | Excluded |
| Settings Change | The settings screen covered the waveform area | Excluded |
| Invalid Range | Could not determine the waveform range | Excluded |
| Empty Waveform | Extracted waveform was empty | Excluded |
| Frame Read Error | Could not read the end frame image | Excluded |
| Axis | The axis labels of some waveform were not read (e.g. a menu over them); that waveform's scale is estimated from VTE / PPEAK and the neighbouring breaths (0.1.6) | Included |

Several values in one breath are separated by ", " (comma and space), e.g. "Abnormal, Fragment". The list is also in outputs.alerts of manifest.json (ENGINE_API.md §8.1).
