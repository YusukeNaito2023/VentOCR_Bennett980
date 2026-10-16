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
