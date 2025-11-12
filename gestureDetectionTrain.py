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
            '0': 'idle',
            '1': "punch",
            '2': "slap",
            '3': "tickle",
            '4': "jitak"
        }
        self.num_classes = len(self.gesture_labels)
        self.input_dim = 70  #60 relative coords + 10 distances

    def create_model(self):
        self.model = tf.keras.Sequential([
            #Input layer
            tf.keras.layers.Input(shape=(self.input_dim,)), 
            tf.keras.layers.Dense(128, activation='relu'),
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
        if x is None or len(x) == 0:
            raise ValueError("No samples to train after preprocessing. Check your CSV and labels.")

        #split data
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        x_val   = self.scaler.transform(x_val)

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
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        gesture_name = self.gesture_labels[str(predicted_class)]
        
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

        # Features numeric & finite
        x = x.astype(np.float32, copy=False)
        invalid_mask = np.any(np.isnan(x) | np.isinf(x), axis=1)
        if np.any(invalid_mask):
            print(f"Warning: Found {invalid_mask.sum()} invalid samples (NaN/Inf). Removing...")
            x = x[~invalid_mask]
            y = y[~invalid_mask]

        # Convert y to numeric -> int for training
        y_num = pd.to_numeric(pd.Series(y), errors='coerce')
        keep_numeric = y_num.notna()
        if not keep_numeric.all():
            print(f"Warning: Found {(~keep_numeric).sum()} non-numeric labels. Removing...")
        x = x[keep_numeric.values]
        y_num = y_num[keep_numeric].astype(int)

        # Filter to allowed classes using string view, but KEEP int dtype for model
        allowed = set(self.gesture_labels.keys())  # {'0','1','2','3','4'}
        y_str = y_num.astype(str)
        keep_allowed = y_str.isin(allowed)
        if not keep_allowed.all():
            print(f"Warning: Found {(~keep_allowed).sum()} invalid labels. Removing...")
        x = x[keep_allowed.values]
        y_int = y_num[keep_allowed].to_numpy(dtype=np.int32)  # <-- int labels for sparse CE

        # Optional: show class counts
        classes, counts = np.unique(y_int, return_counts=True)
        print("Class counts:", dict(zip(classes.tolist(), counts.tolist())))
        print(f"After preprocessing: {len(x)} samples remaining")
        return x, y_int



if __name__ == "__main__":
    gesture_detector = GestureDetector()
    x, y = gesture_detector.load_data("training_data.csv")
    if x is not None and y is not None:
        x, y = gesture_detector.preprocess_data(x, y)
        history = gesture_detector.train(x, y, epochs=50)
        gesture_detector.save_model()
