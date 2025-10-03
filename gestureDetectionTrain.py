import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import glob

class GestureDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.gesture_labels = {
            0: 'idle',
            1: 'move_forward',
            2: 'move_left', 
            3: 'move_backward',
            4: 'move_right'
        }
        self.num_classes = len(self.gesture_labels)
        self.input_dim = 70  #60 relative coords + 10 distances

    def create_model(self):
        self.model = tf.keras.Sequential([
            #Input layer
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            #Layer 1
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            #Layer 2
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            #Output layer
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model
    
    def train(self, x, y, validation_split=0.2, epochs=100, batch_size=32):
        if self.model is None:
            self.create_model()
        
        #normalize features
        x_norm = self.scaler.fit_transform(x)
        
        #split data
        x_train, x_val, y_train, y_val = train_test_split(
            x_norm, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=0.0001
            )
        ]
        
        #train model
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, features):
        if self.model is None:
            print("Model is not loaded or trained.")
            return None, None, None
        
        #reshape and normalize feature
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        #predict gesture
        predictions = self.model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        gesture_name = self.gesture_labels[predicted_class]
        
        return predicted_class, confidence, gesture_name
    
    def evaluate(self, x_test, y_test):
        if self.model is None:
            print("Model is not loaded or trained.")
            return None, None

        x_test_scaled = self.scaler.transform(x_test)
        loss, accuracy = self.model.evaluate(x_test_scaled, y_test, verbose=0)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy

    def save_model(self, model_path="gesture_model.h5", scaler_path="scaler.pkl"):
        if self.model is not None:
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path="gesture_model.h5", scaler_path="scaler.pkl"):
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
            return True
        else:
            print("Model or scaler file not found!")
            return False
        
    def load_data(self, csv_file="training_data.csv"):
        if not os.path.exists(csv_file):
            print(f"Data file {csv_file} not found!")
            return None, None
        
        data = pd.read_csv(csv_file)
        expected_columns = 71
        if len(data.columns) != expected_columns:
            print(f"Data file {csv_file} has incorrect number of columns. Expected {expected_columns}, got {len(data.columns)}")
            return None, None
        
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        return x, y
    
    def preprocess_data(self, x, y):
        print("Preprocessing data...")

        # Check for any invalid values
        if np.any(np.isnan(x)):
            print("Warning: Found NaN values in features. Removing affected samples...")
            valid_mask = ~np.any(np.isnan(x), axis=1)
            x = x[valid_mask]
            y = y[valid_mask]
        
        if np.any(np.isinf(x)):
            print("Warning: Found infinite values in features. Removing affected samples...")
            valid_mask = ~np.any(np.isinf(x), axis=1)
            x = x[valid_mask]
            y = y[valid_mask]
        
        # Ensure labels are in the correct range
        valid_labels = np.isin(y, list(self.gesture_labels.keys()))
        if not np.all(valid_labels):
            print("Warning: Found invalid labels. Removing affected samples...")
            x = x[valid_labels]
            y = y[valid_labels]
        
        print(f"After preprocessing: {len(x)} samples remaining")
        
        return x, y


if __name__ == "__main__":
    gesture_detector = GestureDetector()
    x, y = gesture_detector.load_data("training_data.csv")
    if x is not None and y is not None:
        x, y = gesture_detector.preprocess_data(x, y)
        history = gesture_detector.train(x, y, epochs=50)
        gesture_detector.save_model()
