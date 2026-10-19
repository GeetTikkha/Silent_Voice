import json
import os
import tensorflow as tf

from model import build_model, IMG_SIZE as SIZE

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

IMG_SIZE = (SIZE, SIZE)
BATCH_SIZE = 32
EPOCHS = 30

os.makedirs("models", exist_ok=True)


def load(path, shuffle):
    # batch_size=None -> individual images, so we can cache them before batching
    return tf.keras.utils.image_dataset_from_directory(
        path, image_size=IMG_SIZE, batch_size=None,
        label_mode="categorical", shuffle=shuffle,
    )


train_raw = load(TRAIN_DIR, shuffle=True)
validation_raw = load(VALIDATION_DIR, shuffle=False)

# Save the class order the model was trained with, so webcam/predict can't mismatch.
class_names = train_raw.class_names
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)
print("Classes:", class_names)


def prepare(ds, training):
    # Keep images as uint8 in the cache (4x less RAM). Disk is read only in epoch 1.
    ds = ds.map(lambda x, y: (tf.cast(x, tf.uint8), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if training:
        ds = ds.shuffle(2048)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_dataset = prepare(train_raw, training=True)
validation_dataset = prepare(validation_raw, training=False)

model = build_model(num_classes=len(class_names))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    # Resume support: progress is saved after every epoch in models/backup.
    # Stop with Ctrl+C any time; running train.py again continues from the last finished epoch.
    tf.keras.callbacks.BackupAndRestore("models/backup"),
    # Best model so far is always saved here, so you can test webcam.py before training ends.
    tf.keras.callbacks.ModelCheckpoint(
        "models/isl_cnn.keras", monitor="val_loss", save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
    ),
]

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save("models/isl_cnn.keras")
print("\nTraining completed. Model saved at: models/isl_cnn.keras")
