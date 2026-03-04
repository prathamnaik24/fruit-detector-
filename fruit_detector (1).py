import cv2
import time
import os
from ultralytics import YOLO



WEBCAM_INDEX   = 0  
CONFIDENCE_MIN = 0.45  
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720

FRUIT_CLASSES = {
    46: "banana",
    47: "apple",
    49: "orange",

}

CLASS_COLORS = {
    "banana":   (0,   220, 255),
    "apple":    (0,   180,  60),
    "orange":   (0,   140, 255),

}
DEFAULT_COLOR = (200, 200, 200)

model = YOLO("yolov8n.pt")       
print("[INFO] Model ready.")



cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open webcam at index {WEBCAM_INDEX}. "
        "Try changing WEBCAM_INDEX at the top of the script."
    )

print("[INFO] Webcam opened. Press Q to quit, S to save a screenshot.")



prev_time  = time.time()
screenshot_dir = "screenshots"

def draw_box(frame, label, conf, x1, y1, x2, y2, color):
    """Draw a rounded bounding box + label chip."""
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text    = f"{label}  {conf:.0%}"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = 0.65
    t_thick = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, t_thick)


    pad = 4
    chip_y1 = max(y1 - th - 2 * pad, 0)
    chip_y2 = max(y1, th + 2 * pad)
    cv2.rectangle(frame, (x1, chip_y1), (x1 + tw + 2 * pad, chip_y2), color, -1)
    cv2.putText(frame, text,
                (x1 + pad, chip_y2 - pad - baseline // 2),
                font, scale, (255, 255, 255), t_thick, cv2.LINE_AA)



while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Dropped frame.")
        continue


    results = model(frame, verbose=False)[0]

    fruit_count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in FRUIT_CLASSES:
            continue                 

        conf  = float(box.conf[0])
        if conf < CONFIDENCE_MIN:
            continue

        label = FRUIT_CLASSES[cls_id]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        draw_box(frame, label, conf, x1, y1, x2, y2, color)
        fruit_count += 1

    cur_time  = time.time()
    fps       = 1.0 / max(cur_time - prev_time, 1e-6)
    prev_time = cur_time

    hud = f"FPS: {fps:.1f}   Fruits detected: {fruit_count}"
    cv2.rectangle(frame, (0, 0), (len(hud) * 11 + 10, 32), (0, 0, 0), -1)
    cv2.putText(frame, hud, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)

    cv2.imshow("🍎 Fruit Detector  |  Q = quit  |  S = screenshot", frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        os.makedirs(screenshot_dir, exist_ok=True)
        fname = os.path.join(screenshot_dir, f"fruit_{int(time.time())}.jpg")
        cv2.imwrite(fname, frame)
        print(f"[INFO] Screenshot saved → {fname}")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detector stopped.")
