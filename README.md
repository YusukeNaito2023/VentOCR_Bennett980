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

## Alert（Breaths.csv の Alert 列）

| 値 | 意味 | 精度の評価 |
|---|---|---|
| Fragment | 画面の右端で見切れた呼吸・録画の最初の呼吸（波形の一部だけ） | 除く |
| Abnormal | 呼吸の印（左上のアイコン）が見本と似ていないフレームを含む（設定画面・回路外れなど） | 除く |
| No Sweep | 掃引線が見つからず、波形を切り出せなかった | 除く |
| Settings Change | 設定変更の画面で波形の欄が覆われた | 除く |
| Invalid Range | 波形を切り出す範囲を決められなかった | 除く |
| Empty Waveform | 切り出した波形が空だった | 除く |
| Frame Read Error | 終了フレームの画像を読めなかった | 除く |

1つの呼吸に複数あるときは「, 」（カンマと空白）で区切る（例「Abnormal, Fragment」）。一覧は manifest.json の outputs.alerts にもある（ENGINE_API.md §8.1）。
