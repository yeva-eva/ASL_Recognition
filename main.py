import cv2
import time
import threading
from collections import deque
import mediapipe as mp

from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager
from utils.dataset_utils import load_reference_signs_from_images

# ─── Brightness / Contrast Helpers ──────────────────────────
def apply_clahe(frame):
    """Apply CLAHE to the V channel in HSV for adaptive contrast."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

def apply_gamma(frame, gamma=1.3):
    """Apply a simple gamma correction."""
    inv_g = 1.0 / gamma
    table = (np.linspace(0, 255, 256) / 255.0) ** inv_g * 255.0
    table = table.astype('uint8')
    return cv2.LUT(frame, table)
# ─────────────────────────────────────────────────────────────

# Configuration
SEQ_LEN  = 30        # Number of frames per detection window
COOLDOWN = 1.0       # Seconds to wait between detections
TOP_K    = 3         # Show top K matches

# Shared state
frame_buffer  = deque(maxlen=SEQ_LEN)
detected_sign = ""
last_time     = 0.0
running       = True

def detector_thread_fn(reference_signs):
    """Background thread: runs DTW every SEQ_LEN frames + cooldown."""
    global detected_sign, last_time, running
    recorder = SignRecorder(reference_signs, seq_len=SEQ_LEN)
    while running:
        if len(frame_buffer) == SEQ_LEN and (time.time() - last_time) > COOLDOWN:
            # Prepare recorder
            recorder.reference_signs["distance"].values[:] = 0
            recorder.recorded_results = list(frame_buffer)
            recorder.compute_distances()

            # Pull out top-K matches
            sorted_refs = recorder.reference_signs.sort_values(by="distance").reset_index(drop=True)
            top_matches = sorted_refs.head(TOP_K)
            matches_str = ", ".join(
                f"{row['name']}({row['distance']:.1f})"
                for _, row in top_matches.iterrows()
            )

            # Best candidate
            if not top_matches.empty:
                candidate = top_matches.loc[0, 'name']
                print(f"🔔 Detected sign: {candidate}")
                print(f"   Top {TOP_K}: {matches_str}")
                detected_sign = candidate
                last_time = time.time()
                frame_buffer.clear()
        time.sleep(0.01)

def main():
    global running
    # 1) Build or load reference library once
    reference_signs = load_reference_signs_from_images(max_per_class=500)
    webcam_manager = WebcamManager()

    # 2) Start the detector thread
    t = threading.Thread(
        target=detector_thread_fn,
        args=(reference_signs,),
        daemon=True
    )
    t.start()

    # 3) Main capture + draw loop
    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ─── PREPROCESS ────────────────────────────────
            frame = apply_clahe(frame)                      # normalize contrast
            frame = cv2.GaussianBlur(frame, (5,5), sigmaX=1)  # reduce noise
            # ────────────────────────────────────────────────

            # Run Mediapipe & draw
            image, results = mediapipe_detection(frame, holistic)
            webcam_manager.update(frame, results, detected_sign, is_recording=False)

            # Buffer for background DTW
            frame_buffer.append(results)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 4) Cleanup
    running = False
    t.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
