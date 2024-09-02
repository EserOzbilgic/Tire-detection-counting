from ultralytics import YOLO
import cv2
import numpy as np

# Define paths
model_path = r'yourpath\best.pt'


# Load the YOLOv8 model trained to detect tires
model = YOLO(model_path)

# Open the webcam (or use a video stream if you want to use a different source)
cap = cv2.VideoCapture(0)  # 0 is the default camera; change if using an external camera

if not cap.isOpened():
    raise ValueError("Webcam not found or cannot be opened.")

# Define the color range for tire detection (black color range)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])  # Adjust this range as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frames are returned

    # Perform detection on the current frame
    results = model(frame)

    # Initialize an annotated frame
    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)

            # Crop the detected region
            detected_region = frame[y1:y2, x1:x2]

            # Convert the detected region to HSV color space
            hsv_region = cv2.cvtColor(detected_region, cv2.COLOR_BGR2HSV)

            # Create a mask for the color range
            mask = cv2.inRange(hsv_region, lower_black, upper_black)

            # Calculate the percentage of the detected region that falls within the color range
            percentage_black = (np.sum(mask > 0) / mask.size) * 100

            # Only consider the detection valid if the black percentage is above a certain threshold
            if percentage_black > 50:  # Adjust this threshold based on your needs
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
                cv2.putText(annotated_frame, 'Tire', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label

    # Show the annotated frame
    cv2.imshow('Real-Time Tire Detection', annotated_frame)

    # Break the loop if the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
