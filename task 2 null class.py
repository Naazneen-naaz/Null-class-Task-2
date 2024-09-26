#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from datetime import datetime

# Updated dataset path for emotion detection
dataset_path_train = r'C:\Users\shaik\Desktop\train'
dataset_path_test = r'C:\Users\shaik\Desktop\test'

# CSV file for storing attendance data
attendance_file = r'C:\Users\shaik\Desktop\attendance.csv'

# Initialize CSV data
attendance_data = []

# Time Constraint (9:30 AM to 10:00 AM)
def check_time():
    current_time = datetime.now().time()
    start_time = datetime.strptime("09:30:00", "%H:%M:%S").time()
    end_time = datetime.strptime("10:00:00", "%H:%M:%S").time()
    return start_time <= current_time <= end_time

# Data Generator for Emotion Detection
emotion_datagen = ImageDataGenerator(rescale=1./255)

# Training Data
emotion_train = emotion_datagen.flow_from_directory(
    dataset_path_train,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Testing Data
emotion_test = emotion_datagen.flow_from_directory(
    dataset_path_test,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Number of emotion classes (Happy, Sad, Angry, Neutral, etc.)
num_classes = emotion_train.num_classes

# Build the CNN Model for Emotion Detection
def build_emotion_detection_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer with emotion classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and compile the model
emotion_model = build_emotion_detection_model()
emotion_model.summary()

# Train the model
emotion_model.fit(
    emotion_train,
    epochs=10,  # Adjust the number of epochs
    validation_data=emotion_test
)

# Save the trained emotion detection model with a different filename to avoid conflicts
try:
    emotion_model.save('emotion_detection_model_new.h5')  # New filename to avoid lock issues
    print("Model saved successfully as 'emotion_detection_model_new.h5'")
except OSError as e:
    print(f"Error saving the model: {e}")

# Load Face Detection Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the cascade is loaded correctly
if face_cascade.empty():
    print("Error loading face detection cascade.")
else:
    print("Face detection cascade loaded successfully.")

# Function to mark attendance with detected emotion
def mark_attendance(student_id, emotion):
    current_time = datetime.now().strftime("%H:%M:%S")
    if check_time():  # Only mark attendance if the current time is between 9:30 AM and 10:00 AM
        attendance_data.append({
            "Student_ID": student_id, 
            "Time": current_time, 
            "Status": "Present", 
            "Emotion": emotion
        })

# Save attendance data to CSV
def save_to_file():
    df = pd.DataFrame(attendance_data)
    df.to_csv(attendance_file, index=False)
    print("Attendance data saved to CSV.")

# Real-Time Face Detection and Emotion Detection
def detect_faces_and_emotions():
    cap = cv2.VideoCapture(0)  # Start the webcam

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    student_id_counter = 1  # Simulated student IDs

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract face
            face_img = frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (64, 64))  # Resize to match model input size
            face_img_resized = face_img_resized.reshape(1, 64, 64, 3) / 255.0  # Rescale

            # Predict Emotion
            emotion_prediction = emotion_model.predict(face_img_resized)
            predicted_emotion = emotion_prediction.argmax(axis=-1)
            class_labels = list(emotion_train.class_indices.keys())
            emotion_label = class_labels[predicted_emotion[0]]

            # Simulate student identification (replace with actual recognition system)
            student_id = f"Student_{student_id_counter}"

            # Mark Attendance
            mark_attendance(student_id, emotion_label)

            # Display detected face and emotion on the frame
            cv2.putText(frame, f"ID: {student_id}, Emotion: {emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            student_id_counter += 1

        cv2.imshow('Face and Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run face and emotion detection
print("Starting face and emotion detection.")
detect_faces_and_emotions()

# Save attendance at the end
save_to_file()


# In[ ]:





# In[ ]:




