"""Step 1: crop every dataset image to the hand region.

dataset/original_images/<class>/*.jpg  ->  dataset/cropped_images/<class>/*.png

Detection is tried in 3 passes so fewer images are lost:
  1. direct, 2. with a black border added (helps when the hand fills the frame),
  3. border + lower confidence threshold.
Images already cropped in an earlier run are skipped (safe to re-run).
A few images that still fail are copied to dataset/skipped_samples/ so you can look at them.
"""
import os
import cv2

from hand_utils import HandCropper, IMG_SIZE

SOURCE_DIR = "dataset/original_images"
OUTPUT_DIR = "dataset/cropped_images"
SKIPPED_DIR = "dataset/skipped_samples"
SKIPPED_SAMPLES_PER_CLASS = 5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

strict = HandCropper(static_image_mode=True, min_confidence=0.4)
loose = HandCropper(static_image_mode=True, min_confidence=0.2)


def add_border(image):
    p = int(max(image.shape[:2]) * 0.4)
    return cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def find_crop(image):
    crop, _ = strict.crop(image)
    if crop is not None:
        return crop, "direct"
    padded = add_border(image)
    for label, cropper in (("border", strict), ("border+loose", loose)):
        crop, _ = cropper.crop(padded)
        if crop is not None:
            return crop, label
    return None, None


classes = sorted(
    f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))
)

totals = {"direct": 0, "border": 0, "border+loose": 0, "existing": 0, "skipped": 0}

for class_name in classes:
    src = os.path.join(SOURCE_DIR, class_name)
    dst = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(dst, exist_ok=True)

    counts = {k: 0 for k in totals}
    saved_samples = 0

    for name in os.listdir(src):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        out_path = os.path.join(dst, os.path.splitext(name)[0] + ".png")
        if os.path.exists(out_path):
            counts["existing"] += 1
            continue

        image = cv2.imread(os.path.join(src, name))
        crop, how = (None, None) if image is None else find_crop(image)

        if crop is None:
            counts["skipped"] += 1
            if image is not None and saved_samples < SKIPPED_SAMPLES_PER_CLASS:
                os.makedirs(os.path.join(SKIPPED_DIR, class_name), exist_ok=True)
                cv2.imwrite(os.path.join(SKIPPED_DIR, class_name, name), image)
                saved_samples += 1
            continue

        cv2.imwrite(out_path, cv2.resize(crop, IMG_SIZE))
        counts[how] += 1

    for k in totals:
        totals[k] += counts[k]

    kept = counts["existing"] + counts["direct"] + counts["border"] + counts["border+loose"]
    print(f"{class_name}: kept={kept} (new via border/loose: "
          f"{counts['border'] + counts['border+loose']}), skipped={counts['skipped']}")

kept_total = totals["existing"] + totals["direct"] + totals["border"] + totals["border+loose"]
print(f"\nDone. Kept={kept_total}, Skipped={totals['skipped']}")
print(f"Recovered by border/loose passes this run: {totals['border'] + totals['border+loose']}")
print(f"Look at a few failed images in {SKIPPED_DIR}/ to see why detection fails.")
