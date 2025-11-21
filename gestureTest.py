import cv2
import numpy as np
from handDetection import HandDetector
from gestureDetectionTrain import GestureDetector
import time
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
        
        #load
        if not self.gesture_detector.load_model("gesture_model.h5", "scaler.pkl"):
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
        cap = cv2.VideoCapture(1) #0 for default cam, 1 for droidcam right now
        cap.set(cv2.CAP_PROP_FPS, 60) 
        
        prediction_history = []
        history_size = 5  #number of recent prediction to average
        
        while True:
            success, frame = cap.read()
            if not success:
                continue

            gesture_text = "Gesture: (none)"
            confidence_text = ""
            color = (0, 0, 255)
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.hands.process(rgb_frame)
            now_global = time.time()
            hand_detected = bool(results.multi_hand_landmarks)

            # Handle gap state when hand not detected
            if not hand_detected:
                if not self.in_gap and self.prev_pos is not None:
                    # start a gap window
                    self.in_gap = True
                    self.gap_start_pos = self.prev_pos
                    self.gap_start_time = now_global

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.hand_detector.mpDraw.draw_landmarks(
                        frame, hand_landmarks, self.hand_detector.mpHands.HAND_CONNECTIONS
                    )
                    features = self.hand_detector.getFeatures(hand_landmarks)
                    pred_class, confidence, gesture_name = self.gesture_detector.predict(features)

                    curr_pos = self._centroid(hand_landmarks, frame.shape)
                    now = time.time()
                    inst_speed = 0.0
                    gap_speed_used = False

                    if self.in_gap:
                        gap_dt = now - self.gap_start_time if self.gap_start_time else 0
                        if gap_dt > 0 and gap_dt <= self.max_gap_duration and self.gap_start_pos is not None:
                            dx = curr_pos[0] - self.gap_start_pos[0]
                            dy = curr_pos[1] - self.gap_start_pos[1]
                            disp = math.hypot(dx, dy)
                            inst_speed = disp / gap_dt
                            gap_speed_used = True
                        else:
                            # too long or invalid gap: treat as reset
                            inst_speed = 0.0
                        self.in_gap = False
                        self.gap_start_pos = None
                        self.gap_start_time = None
                    elif self.prev_pos is not None and self.prev_time is not None:
                        dt = now - self.prev_time
                        if dt > 0:
                            dx = curr_pos[0] - self.prev_pos[0]
                            dy = curr_pos[1] - self.prev_pos[1]
                            disp = math.hypot(dx, dy)
                            inst_speed = disp / dt

                    #clamp speed for unrealistic spikes
                    if inst_speed > self.max_plausible_speed:
                        inst_speed = self.max_plausible_speed

                    self.prev_pos = curr_pos
                    self.prev_time = now

                    # smooth speed
                    self.speed_history.append(inst_speed)
                    if len(self.speed_history) > self.speed_history_size:
                        self.speed_history.pop(0)
                    smoothed_speed = sum(self.speed_history) / len(self.speed_history)

                    if pred_class == 0 and smoothed_speed < self.speed_idle_threshold:
                        smoothed_speed = 0.0

                    speed_text = f"Speed: {smoothed_speed:.1f} px/s"
                    if gap_speed_used:
                        speed_text += " (gap avg)"

                    if pred_class is not None:
                        prediction_history.append(pred_class)
                        if len(prediction_history) > history_size:
                            prediction_history.pop(0)
                        if len(prediction_history) >= 3:
                            most_common = max(set(prediction_history), key=prediction_history.count)
                            smoothed_gesture = self.gesture_detector.gesture_labels[str(most_common)]
                        else:
                            smoothed_gesture = gesture_name
                        gesture_text = f"Gesture: {smoothed_gesture}"
                        confidence_text = f"Confidence: {confidence:.2f}"
                        if confidence > 0.8:
                            color = (0, 255, 0)
                        elif confidence > 0.6:
                            color = (0, 255, 255)
                        else:
                            color = (0, 165, 255)
                        gesture_text = f"Gesture: {smoothed_gesture}"

            #display result and confidence
            cv2.putText(frame, gesture_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if confidence_text:
                cv2.putText(frame, confidence_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            #speed display
            if 'speed_text' in locals():
                cv2.putText(frame, speed_text, (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            
            #display window
            cv2.imshow("Gesture Detection Test", frame)
            
            #quit q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #send to unity
            if results.multi_hand_landmarks and pred_class is not None:
                gesture_index = int(pred_class)
                conn.sendall(f"{gesture_index},{smoothed_speed:.2f}\n".encode('utf-8'))
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_system = GestureTestSystem()
    test_system.run_realtime_test()
