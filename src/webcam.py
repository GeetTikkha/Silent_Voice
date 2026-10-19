import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from hand_utils import HandCropper, to_model_input, IMG_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("--mirror", action="store_true",
                    help="flip the frame before predicting (press 'm' to toggle live)")
parser.add_argument("--model", default=None, help="path to a .keras model")
args = parser.parse_args()

# uses the webcam-fine-tuned model automatically if it exists
MODEL_PATH = args.model or (
    "models/isl_cnn_webcam.keras" if os.path.exists("models/isl_cnn_webcam.keras")
    else "models/isl_cnn.keras"
)
CONF_THRESHOLD = 0.60     # below this -> "Uncertain"
SMOOTHING_FRAMES = 8      # average predictions over the last N frames

model = tf.keras.models.load_model(MODEL_PATH)
with open("models/class_names.json") as f:
    class_names = json.load(f)
display_names = [c.replace("_", "/") for c in class_names]
print("Model loaded:", MODEL_PATH, "| Keys: 'q' quit, 'm' toggle mirror.")

cropper = HandCropper(static_image_mode=False)
history = deque(maxlen=SMOOTHING_FRAMES)
mirror = args.mirror

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

blank = np.zeros((256, 256, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    if mirror:
        frame = cv2.flip(frame, 1)

    crop, bbox = cropper.crop(frame)
    model_view = blank.copy()
    top3_lines = []

    if crop is None:
        history.clear()
        text, color = "No hand detected", (0, 0, 255)
    else:
        probs = model(to_model_input(crop), training=False).numpy()[0]
        history.append(probs)
        avg = np.mean(history, axis=0)

        order = np.argsort(avg)[::-1][:3]
        top3_lines = [f"{display_names[i]}: {avg[i] * 100:.0f}%" for i in order]
        idx, conf = int(order[0]), float(avg[order[0]])

        if conf >= CONF_THRESHOLD:
            text, color = f"Sign: {display_names[idx]} ({conf * 100:.0f}%)", (0, 255, 0)
        else:
            text, color = f"Uncertain ({display_names[idx]}? {conf * 100:.0f}%)", (0, 200, 255)

        cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)

        # exactly what the model sees (128x128 crop, shown bigger)
        model_view = cv2.resize(cv2.resize(crop, IMG_SIZE), (256, 256))

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    for i, line in enumerate(top3_lines):
        cv2.putText(frame, line, (20, 75 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"mirror={'on' if mirror else 'off'} (m)  quit (q)",
                (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("ISL Recognition", frame)
    cv2.imshow("Model input (what the CNN sees)", model_view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        mirror = not mirror
        history.clear()

cap.release()
cv2.destroyAllWindows()
