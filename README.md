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

Several values in one breath are separated by ", " (comma and space), e.g. "Abnormal, Fragment". The list is also in outputs.alerts of manifest.json (ENGINE_API.md §8.1).
