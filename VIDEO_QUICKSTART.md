# Video Waste Classification – Quickstart

A fast guide to run the dual‑window video classifier (left: video, right: classification).

## 1) Install

```bash
pip install -r requirements.txt
```

If TensorFlow install fails on Windows, try:
```bash
pip install --upgrade pip
pip install tensorflow
```

## 2) (Optional) Generate Sample Videos

```bash
python create_sample_video.py
```
This will create:
- `sample_videos/waste_sample.mp4`
- `sample_videos/simple_waste.mp4`

## 3) Run the App

```bash
python video_waste_classifier.py
```
Then:
- Click "Load Video" → pick a video (try the ones in `sample_videos/`)
- Click "Play"

## 4) What You’ll See

- Left: actual video with overlays (category + confidence)
- Right: current detection, history, stats, export
- Controls: Play / Pause / Stop + speed slider + progress bar

## 5) Supported Formats

MP4 (recommended), AVI, MOV, MKV, WMV

## 6) Tips for Best Results

- Use clear, well‑lit videos
- Ensure items are large and centered on screen
- Prefer stable footage (minimal motion blur)
- Start with the provided sample videos

## 7) Customize Categories

Edit `waste_categories` in `video_waste_classifier.py`:
```python
self.waste_categories = {
    'plastic': ['plastic bottle', 'plastic bag', 'plastic container'],
    'glass': ['glass bottle', 'glass jar', 'glass container'],
    'metal': ['metal can', 'aluminum', 'tin can'],
    'paper': ['paper', 'cardboard', 'newspaper'],
    'organic': ['food waste', 'vegetable', 'fruit'],
    'wood': ['wooden item', 'wood']
}
```
Adjust confidence threshold in `map_to_waste_category` if needed:
```python
if keyword in label and confidence > 0.3:
```

## 8) Troubleshooting

- Can’t open video: convert to MP4 and try again
- Low accuracy: improve lighting, larger items, simpler background
- Slow performance: lower video resolution; ensure no heavy background apps
- TensorFlow errors on Windows: upgrade pip and reinstall `tensorflow`

## 9) File Map

```
video_waste_classifier.py  # Main app (two windows)
create_sample_video.py     # Generates test videos
sample_videos/             # Output videos
requirements.txt           # Dependencies
VIDEO_README.md            # Full documentation
VIDEO_QUICKSTART.md        # This quickstart
```

You’re ready! Load a sample video and watch the classifications in real time. ♻️
