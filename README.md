# Hand Gesture Detection System

## Overview
This project implements a real-time hand gesture detection system using Machine Learning. It includes modules for data collection, model training, and real-time testing. The system is designed to integrate with a Unity project for interactive visualization.
[Demo Video Link](https://youtu.be/n6rcPzGWFV0)

## Environment Setup

Follow these steps to set up the development environment after cloning the project:

1.  **Create a Virtual Environment**
    ```bash
    python -m venv HandGestureDetection
    ```

2.  **Activate the Virtual Environment**
    ```powershell
    .\HandGestureDetection\Scripts\activate
    ```

3.  **Navigate to the Project Directory**
    ```bash
    cd HandGestureDetection
    ```

4.  **Install Dependencies**
    ```bash
    pip install -r .\requirements.txt
    ```

## Project Components

The project consists of three main Python modules:

### 1. Data Collection
*   **File**: `handDetection.py`
*   **Description**: Used for collecting hand gesture data to build the dataset.

### 2. Model Training
*   **File**: `gestureDetectionTrain.py`
*   **Description**: Handles the training of the machine learning model using the collected data.

### 3. Testing & Inference
*   **File**: `gestureTest.py`
*   **Description**: Runs the real-time gesture recognition to test the trained model. This script also handles the socket connection to Unity.

## Unity Integration

The Unity project is located in the `unity/handGesture` directory.

### Running the Simulation
To run the model with the Unity visualization:

1.  **Start the Python Server**:
    Run the testing script first to establish the socket server.
    ```bash
    python gestureTest.py
    ```
    Wait until you see the message: `Waiting for Unity connection...`

2.  **Launch Unity**:
    *   Open the Unity project found in `unity/handGesture`.
    *   Open the **Final Scene**.
    *   Press **Play** in the Unity Editor.

The Python script will connect to Unity, and detected gestures will be sent to the simulation in real-time.
