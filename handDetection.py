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
            '1': "move_forward",
            '2': "move_left",
            '3': "move_backward",
            '4': "move_right",
        }

        self.current_label = '0'
        self.training_data = []
        self.data_file = 'training_data.csv'

        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [f'feature_{i}' for i in range(70)] + ['label']
                writer.writerow(headers)
    
    def getFeatures(self, hand_landmarks):
        features = []
        landmarks = []

        #get coordinates of each hand landmarks
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        #normalize the coordinates relative to wrist
        wrist = landmarks[0]
        for i in range(1, len(landmarks)):
            relative_x = landmarks[i][0] - wrist[0]
            relative_y = landmarks[i][1] - wrist[1]
            relative_z = landmarks[i][2] - wrist[2]
            features.extend([relative_x, relative_y, relative_z])

        #get distance between fingertips
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        for i, tip1 in enumerate(finger_tips):
            for tip2 in range(i + 1, len(finger_tips)):
                dist = np.sqrt((landmarks[tip1][0] - landmarks[tip2][0])**2 +
                               (landmarks[tip1][1] - landmarks[tip2][1])**2 + 
                               (landmarks[tip1][2] - landmarks[tip2][2])**2)
                features.append(dist)

        #feature array will be filled with coordinates of each landmark (3 per landmark)
        #relative to the wrist and distances between fingertips
        #total elements will be 60 (relative coords) + 10 (distances) = 70
        return np.array(features)
    
    def run(self):
        pTime = 0

        while True:
            success, img = self.videoCapture.read()
            if not success:
                continue
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            current_features = None
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    features = self.getFeatures(handLms)
                    #print(features)

            # get fps
            cTime = time.time()
            fps = 1 / (cTime - pTime) if cTime != pTime else 0
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            #display window
            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF

            #0, 1, 2, 3, 4, to select gesture
            #s to save the feature and label it
            #q to quit
            if(key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]):
                self.current_label = chr(key)
                self.current_features = features if results.multi_hand_landmarks else None
                if self.current_features is None:
                    print("No hand detected try again")
                print(f"Selected gesture: {self.gesture_labels[self.current_label]}")
            elif key == ord('s'):
                if self.current_features is not None:
                    self.save_training_sample(self.current_features, self.current_label)
                else:
                    print("No hand detected! Cannot save sample.")
            elif key == ord('q'):
                break

        self.videoCapture.release()
        cv2.destroyAllWindows()

    def save_training_sample(self, features, label):
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = features.tolist() + [label]
            writer.writerow(row)
        print(f"Saved sample with label '{self.gesture_labels[label]}' ({label})")



if __name__ == '__main__':
   handDetector = HandDetector()
   handDetector.run()