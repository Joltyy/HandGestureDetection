import cv2
import numpy as np
from handDetection import HandDetector
from gestureDetectionTrain import GestureDetector
import time
from collections import deque
import socket
import json
import math

# HOST = '127.0.0.1'
# PORT = 5005

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((HOST, PORT))
# s.listen(1)

# print("Waiting for Unity connection...")
# conn, addr = s.accept()
# print(f"Connected by {addr}")

class GestureTestSystem:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.gesture_detector = GestureDetector()
        
        #allowing motion to capture state for punch
        self.t_window = 16 #number of frames to consider
        self.feature_window = deque(maxlen=self.t_window) #double-ended queue to hold features

        #hysteresis variables for punch detection
        self.punch_on_count = 0
        self.punch_off_count = 0
        self.punch_active = False
        self.last_bbox_area = None #last bounding box area for punch detection

        # thresholds – tune live if needed
        self.THRUST_DIFF_HIGH = 0.025       # enter threshold
        self.THRUST_DIFF_LOW  = 0.015       # stay threshold
        self.AREA_GROWTH_HIGH = 0.10        # +10% area growth to enter
        self.AREA_GROWTH_LOW  = 0.05        # +5% area growth to stay
        self.ENTER_FRAMES = 2               # consecutive frames to enter
        self.EXIT_FRAMES  = 3               # consecutive frames to exit


        #load
        if not self.gesture_detector.load_model("gesture_model.keras", "scaler.pkl"):
            raise Exception("Failed to load trained model!")
        print("Model loaded successfully!")
        
        self.prev_pos = None          # last stable centroid (x,y)
        self.prev_time = None
        self.speed_history = []
        self.speed_history_size = 5
        self.speed_idle_threshold = 15.0    # pixels/sec threshold to consider idle speed = 0

        # gap tracking
        self.in_gap = False
        self.gap_start_pos = None
        self.gap_start_time = None
        self.max_gap_duration = 1.0   # ignore if gap exceeds (treat as reset)
        self.max_plausible_speed = 2500.0  # px/s clamp
    
    def _centroid(self, hand_landmarks, frame_shape):
        h, w = frame_shape[:2]
        # use finger tips + wrist for robustness
        tip_indices = [0, 4, 8, 12, 16, 20]
        pts = []
        for i in tip_indices:
            lm = hand_landmarks.landmark[i]
            pts.append((lm.x * w, lm.y * h))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    def run_realtime_test(self):
        cap = cv2.VideoCapture(0)

        prev_wrist_px = None
        prev_time = None
        speed_threshold = 30.0  # pixels/sec — adjust to your setup

        prediction_history = []
        history_size = 5  # number of recent predictions to average

        while True:
            success, frame = cap.read()
            if not success:
                continue

            now = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.hands.process(rgb_frame)

            gesture_text = "No hand detected"
            confidence_text = ""
            color = (0, 0, 255)
            speed = 0.0
            moving = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # draw landmarks
                    self.hand_detector.mpDraw.draw_landmarks(
                        frame, hand_landmarks, self.hand_detector.mpHands.HAND_CONNECTIONS
                    )

                    # compute wrist pixel position and speed
                    h, w = frame.shape[:2]
                    wrist = hand_landmarks.landmark[0]
                    wrist_px = np.array([wrist.x * w, wrist.y * h])

                    if prev_wrist_px is not None and prev_time is not None:
                        dt = now - prev_time
                        if dt > 0:
                            dist = np.linalg.norm(wrist_px - prev_wrist_px)
                            speed = dist / dt  # pixels per second
                            moving = speed >= speed_threshold
                    # DEBUG: log raw wrist speed computation
                    # (prints every frame a hand is detected; useful to verify values)
                    try:
                        print(f"[DEBUG] wrist_px={wrist_px} dt={dt:.4f} speed={speed:.2f} moving={moving}")
                    except Exception:
                        # dt may be undefined on first frame; ignore
                        pass
                    prev_wrist_px = wrist_px
                    prev_time = now

                    # predict gesture
                    features = self.hand_detector.getFeatures(hand_landmarks)
                    pred_class, confidence, gesture_name = self.gesture_detector.predict(features)

                    if pred_class is not None:
                        # add to history
                        prediction_history.append(pred_class)
                        if len(prediction_history) > history_size:
                            prediction_history.pop(0)

                        # smoothing
                        if len(prediction_history) >= 3:
                            most_common = max(set(prediction_history), key=prediction_history.count)
                            smoothed_gesture = self.gesture_detector.gesture_labels[str(most_common)]
                        else:
                            smoothed_gesture = gesture_name

                        state = "moving" if moving else "stationary"
                        speed_text = f"Speed: {speed:.1f} px/s ({state})"

                        # show speed info for all gestures (helps debugging)
                        state = "moving" if moving else "stationary"
                        speed_text = f"Speed: {speed:.1f} px/s ({state})"

                        gesture_text = f"Gesture: {smoothed_gesture}"
                        confidence_text = f"Confidence: {confidence:.2f}"

                        # # send to unity: "index,speed\n" (speed is 0 when stationary)
                        # gesture_index = int(pred_class)
                        # # DEBUG: print what we send to Unity
                        # try:
                        #     print(f"[DEBUG] Sending to Unity: index={gesture_index}, speed={speed:.2f}, gesture={smoothed_gesture}")
                        # except Exception:
                        #     pass
                        # conn.sendall(f"{gesture_index},{speed:.2f}\n".encode('utf-8'))

                    # display result and confidence on frame
                    cv2.putText(frame, gesture_text, (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if confidence_text:
                        cv2.putText(frame, confidence_text, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    # draw speed_text for visibility (only if non-empty)
                    if speed_text:
                        cv2.putText(frame, speed_text, (10, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Gesture Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_system = GestureTestSystem()
    test_system.run_realtime_test()
