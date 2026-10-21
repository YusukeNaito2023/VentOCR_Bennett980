ENGINE_SETTINGS_SCHEMA = {
    "initial_ocr": {
        "skip_keys": {
            "type": "multiselect",
            "choices": ["Paw", "Flow", "Volume"],
            "default": [],
            "label": "ROI Keys to be skipped"
        },
        "sampling_size": {
            "type": "int",
            "default": 10,
            "label": "Sampling Interval (Frames)"
        },
        "use_gpu": {
            "type": "bool",
            "default": True,
            "label": "Enable GPU Processing"
        }
    },
    "post_ocr": {
        "fps": {
            "type": "int",
            "default": 10,
            "label": "Frame Rate (FPS)"
        },
        "trim_to_breath": {
            "type": "bool",
            "default": True,
            "label": "Align waveform boundaries to inspiration onset"
        },
        "smoothing": {
            "type": "bool",
            "default": False,
            "label": "Smooth waveforms (moving average)"
        },
        "imputation": {
            "type": "bool",
            "default": False,
            "label": "Fill undetected waveform points by linear interpolation"
        }
    }
}
