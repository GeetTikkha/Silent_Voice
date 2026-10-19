import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from model import IMG_SIZE as SIZE

IMG_SIZE = (SIZE, SIZE)

model = tf.keras.models.load_model("models/isl_cnn.keras")
with open("models/class_names.json") as f:
    class_names = json.load(f)

ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test", image_size=IMG_SIZE, batch_size=32,
    label_mode="int", shuffle=False, class_names=class_names)

y_true = np.concatenate([y.numpy() for _, y in ds])
y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)

print(f"\nTest accuracy: {np.mean(y_true == y_pred) * 100:.2f}%  ({len(y_true)} images)\n")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
