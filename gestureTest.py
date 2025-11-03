import cv2
import numpy as np
from handDetection import HandDetector
from gestureDetectionTrain import GestureDetector
import time

class GestureTestSystem:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.gesture_detector = GestureDetector()
        
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