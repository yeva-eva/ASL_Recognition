import cv2
import os
import numpy as np
import pickle as pkl
import mediapipe as mp
from utils.mediapipe_utils import mediapipe_detection


def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    """
    Returns three lists (pose, left_hand, right_hand), each
    fixed-length, zero-filled if Mediapipe didn’t detect that part.
    """
    def lm_to_list(mp_landmarks, n_coords):
        if mp_landmarks is not None:
            return landmark_to_array(mp_landmarks).reshape(n_coords).tolist()
        else:
            return [0] * n_coords

    # pose: 33 keypoints × (x,y,z)
    pose      = lm_to_list(results.pose_landmarks,      33 * 3)
    # each hand: 21 keypoints × (x,y,z)
    left_hand = lm_to_list(results.left_hand_landmarks,  21 * 3)
    right_hand= lm_to_list(results.right_hand_landmarks, 21 * 3)

    return pose, left_hand, right_hand


def save_landmarks_from_video(video_name):
    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}
    sign_name = video_name.split("-")[0]

    # Set the Video stream
    cap = cv2.VideoCapture(
        os.path.join("data", "videos", sign_name, video_name + ".mp4")
    )
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Store results
                pose, left_hand, right_hand = extract_landmarks(results)
                landmark_list["pose"].append(pose)
                landmark_list["left_hand"].append(left_hand)
                landmark_list["right_hand"].append(right_hand)
            else:
                break
        cap.release()

    # Create the folder of the sign if it doesn't exists
    path = os.path.join("data", "dataset", sign_name)
    if not os.path.exists(path):
        os.mkdir(path)

    # Create the folder of the video data if it doesn't exists
    data_path = os.path.join(path, video_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Saving the landmark_list in the correct folder
    save_array(
        landmark_list["pose"], os.path.join(data_path, f"pose_{video_name}.pickle")
    )
    save_array(
        landmark_list["left_hand"], os.path.join(data_path, f"lh_{video_name}.pickle")
    )
    save_array(
        landmark_list["right_hand"], os.path.join(data_path, f"rh_{video_name}.pickle")
    )


def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)
