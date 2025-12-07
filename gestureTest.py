import cv2
import numpy as np
from handDetection import HandDetector
from gestureDetectionTrain import GestureDetector
import time
from collections import deque
import socket
import json
import math

HOST = '127.0.0.1'
PORT = 5005

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

print("Waiting for Unity connection...")
conn, addr = s.accept()
print(f"Connected by {addr}")

class GestureTestSystem:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.gesture_detector = GestureDetector()
        self.last_detect_time = None
        self.reset_gap_seconds = 3.0
        
        #load
        if not self.gesture_detector.load_model("gesture_model.keras", "scaler.pkl"):
            raise Exception("Failed to load trained model!")
        print("Model loaded successfully!")
        
    def run_realtime_test(self):
        cap = cv2.VideoCapture(0)

        prev_landmarks_px = None
        prev_time = None
        speed_threshold = 100.0  # pixels/sec

        prediction_history = []
        history_size = 6  # number of recent predictions to average

        while True:
            success, frame = cap.read()
            if not success:
                continue

            now = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.hands.process(rgb_frame)

            if self.last_detect_time is not None and not results.multi_hand_landmarks:
                if (now - self.last_detect_time) >= self.reset_gap_seconds:
                    prev_landmarks_px = None
                    prev_time = None

            gesture_text = "No hand detected"
            confidence_text = ""
            color = (0, 0, 255)
            speed = 0.0
            moving = False
            speed_text = ""
            hand_detected_flag = 1 if results.multi_hand_landmarks else 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # draw landmark
                    self.hand_detector.mpDraw.draw_landmarks(
                        frame, hand_landmarks, self.hand_detector.mpHands.HAND_CONNECTIONS
                    )

                    # compute wrist pixel position and speed
                    h, w = frame.shape[:2]
                    curr_landmarks_px = np.array(
                        [[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark], dtype=np.float32
                    )

                    if prev_landmarks_px is not None and prev_time is not None:
                        dt = now - prev_time
                        if dt > 0:
                            disp = np.linalg.norm(curr_landmarks_px - prev_landmarks_px, axis=1)  #
                            mean_disp = float(np.mean(disp)) #average displacement of all points
                            speed = mean_disp / dt
                            moving = speed >= speed_threshold

                    prev_landmarks_px = curr_landmarks_px
                    prev_time = now
                    self.last_detect_time = now

                    # predict gesture
                    features = self.hand_detector.getFeatures(hand_landmarks)
                    pred_class, confidence, gesture_name = self.gesture_detector.predict(features)

                    if pred_class is not None:
                        # add to history
                        prediction_history.append(pred_class)
                        if len(prediction_history) > history_size:
                            prediction_history.pop(0)

                        # smoothing
                        most_common = 0
                        if len(prediction_history) >= 3:
                            most_common = max(set(prediction_history), key=prediction_history.count)
                            smoothed_gesture = self.gesture_detector.gesture_labels[str(most_common)]
                        else:
                            smoothed_gesture = gesture_name

                        state = "moving" if moving else "stationary"

                        speed_text = f"Speed: {speed:.1f} px/s ({state})"
                        gesture_text = f"Gesture: {smoothed_gesture}"
                        confidence_text = f"Confidence: {confidence:.2f}"

                        #send to unity
                        gesture_index = int(most_common)
                        try:
                            print(f"[DEBUG] Sending to Unity: index={gesture_index}, speed={speed:.2f}, gesture={smoothed_gesture}")
                        except Exception:
                            pass
                    
                    # display result
                    cv2.putText(frame, gesture_text, (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, confidence_text, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, speed_text, (10, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                gesture_index = 0
                speed = 0.0
                gesture_text = "No hand detected"
                confidence_text = ""
                speed_text = "Speed: 0.0 px/s (stationary)"

            conn.sendall(f"{gesture_index},{speed:.2f}, {hand_detected_flag}\n".encode('utf-8'))
            cv2.imshow("Gesture Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_system = GestureTestSystem()
    test_system.run_realtime_test()
