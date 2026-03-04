# 🍎 Real-Time Fruit Detector

Detects fruits live from your webcam using **YOLOv8** + **OpenCV**.  
Draws a labeled bounding box around every fruit it spots.

## Detected fruits
banana · apple · orange · broccoli · carrot · hot dog · pizza · donut · cake · sandwich

---

## Quick setup (3 steps)

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads the tiny YOLOv8n model (~6 MB) automatically.

### 2 — Run
```bash
python fruit_detector.py
```

### 3 — Controls
| Key | Action |
|-----|--------|
| **Q** | Quit |
| **S** | Save screenshot to `screenshots/` folder |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Webcam not found | Change `WEBCAM_INDEX = 0` → `1` or `2` at the top of `fruit_detector.py` |
| Low FPS | Lower `FRAME_WIDTH`/`FRAME_HEIGHT` in the config section |
| Too many false positives | Raise `CONFIDENCE_MIN` (e.g. `0.60`) |
| Missing detections | Lower `CONFIDENCE_MIN` (e.g. `0.35`) |

---

## How it works

1. **YOLOv8n** runs inference on every webcam frame.
2. Detections are filtered to fruit/food COCO class IDs only.
3. OpenCV draws a colour-coded rectangle + label chip for each hit.
4. FPS and fruit count are shown in the top-left HUD.
