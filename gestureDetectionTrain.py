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
        }
        self.num_classes = len(self.gesture_labels)
        self.input_dim = 70  #60 relative coords + 10 distances

    def create_model(self):
        self.model = tf.keras.Sequential([
            #Input layer
            tf.keras.layers.Input(shape=(self.input_dim,)), 

            #Layer 1
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            #Layer 2
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            #Output layer
            tf.keras.layers.Dense(self.num_classes)
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(weight_decay=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
            metrics=["accuracy"]
        )

        return self.model
    
    def train(self, x, y, validation_split=0.2, epochs=100, batch_size=16):
        if self.model is None:
            self.create_model()
        if x is None or len(x) == 0:
            raise ValueError("No samples to train after preprocessing. Check your CSV and labels.")

        #split data
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=validation_split, random_state=0, stratify=y
        )
        
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        x_val   = self.scaler.transform(x_val)

        # Callbacks
        callbacks = [
            # tf.keras.callbacks.EarlyStopping(
            #     monitor='val_loss', 
            #     patience=20, 
            #     restore_best_weights=True
            # ),
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

        train_acc = history.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', [])
        for i, (a, va) in enumerate(zip(train_acc, val_acc), start=1):
            print(f"Epoch {i}/{len(train_acc)} — train_acc={a:.4f}  val_acc={va:.4f}")
        
        if train_acc:
            print(f"Final — train_acc={train_acc[-1]:.4f}  val_acc={val_acc[-1]:.4f}")

        return history
    
    def predict(self, features):
        if self.model is None:
            print("Model is not loaded or trained.")
            return None, None, None
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        logits = self.model.predict(features_scaled, verbose=0)[0]
        probs = tf.nn.softmax(logits).numpy()
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
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

    def save_model(self, model_path="gesture_model.keras", scaler_path="scaler.pkl"):
        if self.model is not None:
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path="gesture_model.keras", scaler_path="scaler.pkl"):
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
        x = x.astype(np.float32, copy=False)
        invalid_mask = np.any(np.isnan(x) | np.isinf(x), axis=1)
        if np.any(invalid_mask):
            print(f"Found {invalid_mask.sum()} invalid samples (NaN/Inf)")
            x = x[~invalid_mask]
            y = y[~invalid_mask]

        y_num = pd.to_numeric(pd.Series(y), errors='coerce')
        keep_numeric = y_num.notna()
        if not keep_numeric.all():
            print(f"Found {(~keep_numeric).sum()} non-numeric labels")
        x = x[keep_numeric.values]
        y_num = y_num[keep_numeric].astype(int)

        #check labels
        allowed = set(self.gesture_labels.keys())
        y_str = y_num.astype(str)
        keep_allowed = y_str.isin(allowed)
        if not keep_allowed.all():
            print(f"Found {(~keep_allowed).sum()} invalid labels")
        x = x[keep_allowed.values]
        y_int = y_num[keep_allowed].to_numpy(dtype=np.int32)

        #y_onehot = tf.keras.utils.to_categorical(y_int, num_classes=self.num_classes)

        return x, y_int



if __name__ == "__main__":
    gesture_detector = GestureDetector()
    x, y = gesture_detector.load_data("training_data.csv")
    if x is not None and y is not None:
        x, y = gesture_detector.preprocess_data(x, y)
        history = gesture_detector.train(x, y, epochs=50)
        gesture_detector.save_model()
