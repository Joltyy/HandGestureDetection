import cv2
import mediapipe as mp
import time
import numpy as np
import csv
import os
from collections import Counter

class HandDetector:
    def __init__(self):
        self.videoCapture = cv2.VideoCapture(0)
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.gesture_labels = {
            '0': 'idle',
            '1': "punch",
            '2': "slap",
            '3': "tickle",
        }

        self.current_label = '0'
        self.training_data = []
        self.data_file = 'training_data.csv'
        self.current_features = None

        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [f'feature_{i}' for i in range(70)] + ['label']
                writer.writerow(headers)
    
    def getFeatures(self, hand_landmarks):
        features = []
        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        wrist = landmarks[0]
        for i in range(1, len(landmarks)):
            relative_x = landmarks[i][0] - wrist[0]
            relative_y = landmarks[i][1] - wrist[1]
            relative_z = landmarks[i][2] - wrist[2]
            features.extend([relative_x, relative_y, relative_z])

        finger_tips = [4, 8, 12, 16, 20]
        for i, tip1 in enumerate(finger_tips):
            for tip2 in range(i + 1, len(finger_tips)):
                # 2D distance (xy)
                dist = np.sqrt((landmarks[tip1][0] - landmarks[tip2][0])**2 +
                               (landmarks[tip1][1] - landmarks[tip2][1])**2)
                features.append(dist)

        # 60 rel coords + 10 distances = 70
        return np.array(features)
    
    def run(self):
        pTime = 0
        features = None  # ensure defined every loop

        while True:
            success, img = self.videoCapture.read()
            if not success:
                continue
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            features = None
            if results.multi_hand_landmarks:
                # use the first hand only for consistency
                handLms = results.multi_hand_landmarks[0]
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                features = self.getFeatures(handLms)

            # get fps
            cTime = time.time()
            fps = 1 / (cTime - pTime) if cTime != pTime else 0
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            #display window
            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF

            #0, 1, 2, 3, 4, 5 to select gesture
            #s to save the feature and label it
            #q to quit

            if key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                self.current_label = chr(key)
                if features is not None:
                    self.current_features = features
                    print(f"Selected gesture: {self.gesture_labels[self.current_label]}")
                else:
                    self.current_features = None
                    print("No hand detected! Cannot select gesture.")

            elif key == ord('s'):
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(imgRGB)
                if results.multi_hand_landmarks:
                    handLms = results.multi_hand_landmarks[0]
                    fresh_features = self.getFeatures(handLms)
                else:
                    fresh_features = None

                if fresh_features is not None and self.current_label in self.gesture_labels:
                    self.save_training_sample(fresh_features, self.current_label)
                else:
                    print("No hand detected! Cannot save sample.")

            elif key == ord('q'):
                break

        self.videoCapture.release()
        cv2.destroyAllWindows()

    def save_training_sample(self, features, label):
        if features is None or len(features) != 70 or np.isnan(features).any() or np.isinf(features).any():
            print("Skipped saving: invalid feature vector.")
            return
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = features.tolist() + [label]
            writer.writerow(row)
        print(f"Saved sample with label '{self.gesture_labels[label]}' ({label})")



if __name__ == '__main__':
   handDetector = HandDetector()
   handDetector.run()