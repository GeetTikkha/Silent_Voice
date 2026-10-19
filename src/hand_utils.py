"""Shared hand-cropping + preprocessing.

The SAME code is used for (1) cropping the dataset, (2) webcam collection,
(3) live prediction, so training and live inputs look identical.
"""
import cv2
import numpy as np
import mediapipe as mp

IMG_SIZE = (128, 128)


class HandCropper:
    """Finds hand(s) with MediaPipe and returns a square crop around them.

    All detected hands (max 2) are included, since many ISL signs use both.
    """

    def __init__(self, static_image_mode=False, padding=0.25, min_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )
        self.padding = padding

    def crop(self, frame_bgr):
        """Returns (crop_bgr, bbox) or (None, None) if no hand is found.
        bbox = (x1, y1, x2, y2) in frame coordinates (may extend past edges)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, None

        xs = [lm.x * w for hand in result.multi_hand_landmarks for lm in hand.landmark]
        ys = [lm.y * h for hand in result.multi_hand_landmarks for lm in hand.landmark]

        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * self.padding)
        half = max(int(side) // 2, 16)
        side = half * 2

        # pad the frame with black so crops near the edge stay square
        padded = cv2.copyMakeBorder(
            frame_bgr, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        x1, y1 = int(cx) - half, int(cy) - half
        crop = padded[y1 + half : y1 + half + side, x1 + half : x1 + half + side]
        return crop, (x1, y1, x1 + side, y1 + side)

    def close(self):
        self.hands.close()


def to_model_input(crop_bgr):
    """BGR crop -> (1, 128, 128, 3) float32 in 0..255.
    Rescaling to 0..1 happens INSIDE the model, so don't divide by 255 here."""
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return np.expand_dims(img.astype(np.float32), axis=0)
