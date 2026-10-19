import json
import sys
import cv2
import numpy as np
import tensorflow as tf

from hand_utils import HandCropper, to_model_input

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

model = tf.keras.models.load_model("models/isl_cnn.keras")
with open("models/class_names.json") as f:
    class_names = json.load(f)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise SystemExit("Error: Image could not be loaded.")

crop, _ = HandCropper(static_image_mode=True).crop(image)
if crop is None:
    raise SystemExit("No hand detected in the image.")

probs = model.predict(to_model_input(crop), verbose=0)[0]
top = np.argsort(probs)[::-1][:3]

print("\nPREDICTION RESULT")
for i in top:
    print(f"{class_names[i]}: {probs[i] * 100:.2f}%")
