"""Step 2: split cropped images 80/10/10 (train/validation/test)."""
import os
import random
import shutil

SOURCE_DIR = "dataset/cropped_images"   # <- now the cropped images
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10

random.seed(42)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# remove old splits so stale files from earlier runs don't linger
for split in ("train", "validation", "test"):
    shutil.rmtree(os.path.join(OUTPUT_DIR, split), ignore_errors=True)

classes = sorted(
    f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))
)
print(f"Total classes found: {len(classes)}")

for class_name in classes:
    src = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXTENSIONS)]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VALIDATION_RATIO)

    splits = {
        "train": images[:n_train],
        "validation": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, files in splits.items():
        dst = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    print(f"{class_name}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

print("\nDataset splitting completed successfully!")
