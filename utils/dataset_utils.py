import os, glob
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm

from models.sign_model import SignModel
from utils.landmark_utils import landmark_to_array

def load_reference_signs_from_images(
    dataset_dir: str = os.path.join("data", "dataset"),
    min_conf: float = 0.5,
    max_per_class: int = 500,
):
    # 1) Gather up to max_per_class images per letter
    samples = []
    for letter in sorted(os.listdir(dataset_dir)):
        letter_dir = os.path.join(dataset_dir, letter)
        if not os.path.isdir(letter_dir):
            continue
        all_imgs = glob.glob(os.path.join(letter_dir, "*.jpg"))
        for img_path in all_imgs[:max_per_class]:
            samples.append((letter, img_path))

    data = {"name": [], "sign_model": [], "distance": []}

    # 2) Process with Mediapipe Hands
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_conf,
    ) as hands:

        for letter, img_path in tqdm(samples, desc="Building reference library"):
            img = cv2.imread(img_path)
            # mirror it so a left-hand photo turns into a right-hand shape
            img = cv2.flip(img, 1)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Prepare defaults
            left_landmarks  = [0]*(21*3)
            right_landmarks = [0]*(21*3)

            # Fill in if detected
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                                 results.multi_handedness):
                    arr = landmark_to_array(hand_lms).reshape(21*3).tolist()
                    side = handedness.classification[0].label
                    if side == "Left":
                        left_landmarks = arr
                    else:
                        right_landmarks = arr

            # Skip if no hand
            if sum(left_landmarks) + sum(right_landmarks) == 0:
                continue

            data["name"].append(letter)
            data["sign_model"].append(SignModel([left_landmarks], [right_landmarks]))
            data["distance"].append(0)

    df = pd.DataFrame(data, dtype=object)
    print("\nLoaded reference counts:\n", df.groupby("name")["sign_model"].count())
    return df
