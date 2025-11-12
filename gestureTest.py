import cv2
import numpy as np
from handDetection import HandDetector
from gestureDetectionTrain import GestureDetector
import time
from collections import deque

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
        if not self.gesture_detector.load_model("gesture_model.h5", "scaler.pkl"):
            raise Exception("Failed to load trained model!")
        
        print("Model loaded successfully!")
    
    def run_realtime_test(self):
        cap = cv2.VideoCapture(0)
        
        prediction_history = []
        history_size = 5  #number of recent prediction to average
        
        while True:
            success, frame = cap.read()
            if not success:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.hands.process(rgb_frame)
            
            gesture_text = "No hand detected"
            confidence_text = ""
            color = (0, 0, 255)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #draw landmarks
                    self.hand_detector.mpDraw.draw_landmarks(
                        frame, hand_landmarks, self.hand_detector.mpHands.HAND_CONNECTIONS
                    )
                    features = self.hand_detector.getFeatures(hand_landmarks)
                    
                    self.feature_window.append(features) #push into sliding window

                    curr_d = features[-10:]
                    thrust_score = 0.0
                    if len(self.feature_window) >= 2:
                        prev_d = self.feature_window[-2][-10:]
                        thrust_score = float(np.mean(np.abs(curr_d - prev_d)))

                    try:
                        xs = [lm.x for lm in hand_landmarks.landmark]
                        ys = [lm.y for lm in hand_landmarks.landmark]
                        w = (max(xs) - min(xs))
                        h = (max(ys) - min(ys))
                    except:
                        bbox_area = None

                    area_growth = 0.0
                    if self.last_bbox_area is not None and bbox_area is not None:
                        area_growth = (bbox_area - self.last_bbox_area) / self.last_bbox_area
                    if bbox_area is not None:
                        self.last_bbox_area = bbox_area


                    #predict gesture
                    pred_class, confidence, gesture_name = self.gesture_detector.predict(features)
                    
                    if pred_class is not None:
                        #add to history
                        prediction_history.append(pred_class)
                        if len(prediction_history) > history_size:
                            prediction_history.pop(0)
                        
                        #get the average prediction from current history
                        if len(prediction_history) >= 3:
                            most_common = max(set(prediction_history), key=prediction_history.count)
                            smoothed_gesture = self.gesture_detector.gesture_labels[str(most_common)]
                        else:
                            smoothed_gesture = gesture_name
                        
                        #punch gate with motion + hysteresis
                        looks_like_punch = (smoothed_gesture == "punch") and confidence > 0.6
                        enter_motion = (thrust_score > self.THRUST_DIFF_HIGH) or (area_growth > self.AREA_GROWTH_HIGH)
                        stay_motion = (thrust_score > self.THRUST_DIFF_LOW) or (area_growth > self.AREA_GROWTH_LOW)

                        wants_on = looks_like_punch and (enter_motion or (self.punch_active and stay_motion))

                        if wants_on:
                            self.punch_on_count = min(self.punch_on_count + 1, self.ENTER_FRAMES)
                            self.punch_off_count = 0
                        else:
                            self.punch_off_count += 1
                            if self.punch_off_count >= self.EXIT_FRAMES:
                                self.punch_on_count = 0

                        self.punch_active = (self.punch_on_count >= self.ENTER_FRAMES)

                        gesture_text = f"Gesture: {smoothed_gesture}"
                        confidence_text = f"Confidence: {confidence:.2f}"
                        
                        if confidence > 0.8:
                            color = (0, 255, 0) #green
                        elif confidence > 0.6:
                            color = (0, 255, 255) #yellow
                        else:
                            color = (0, 165, 255) #orange
            
            #display result and confidence
            cv2.putText(frame, gesture_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if confidence_text:
                cv2.putText(frame, confidence_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            
            #display window
            cv2.imshow("Gesture Detection Test", frame)
            
            #quit q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        test_system = GestureTestSystem()
        test_system.run_realtime_test()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained the model first by running gestureDetection.py")