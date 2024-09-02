from ultralytics import YOLO
import cv2
import numpy as np

# Define paths
model_path = r'yourpath\best.pt'


# Load the YOLOv8 model trained to detect tires
model = YOLO(model_path)

# RTSP URL with authentication (replace 'username' and 'password' with your actual credentials)
rtsp_url = 'rtsp://username:password@ip/ONVIF/yourprofile'

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    raise ValueError(f"RTSP stream not found or cannot be opened: {rtsp_url}")

# Get class names and print them
class_names = {0: 'tire'}
print("Available class names:", class_names)

# Check if 'tire' is in the class names
if 'tire' not in class_names.values():
    raise ValueError("'tire' class not found in the model classes. Available classes are:", class_names)

# Get the index of 'tire'
tire_class_index = list(class_names.keys())[list(class_names.values()).index('tire')]

# Define a function to detect contours within the detected tire bounding boxes
def detect_contours(frame, tire_boxes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for box in tire_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        roi = gray[y1:y2, x1:x2]
        _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours within the bounding box
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Adjust the area threshold based on your needs
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'Tire', (x1 + x, y1 + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frames are returned

    # Perform detection on the current frame with a lower confidence threshold to detect closely packed tires
    results = model(frame, conf=0.25)  # Adjust confidence threshold if needed

    # Filter out only tire detections
    detections = results[0]
    tire_boxes = [box for box in detections.boxes if int(box.cls) == tire_class_index]

    # Create a copy of the frame to draw the annotations
    annotated_frame = frame.copy()

    # Draw bounding boxes for detected tires
    for box in tire_boxes:
        # Convert tensor coordinates to a list and then to integers
        coords = box.xyxy.tolist()[0]
        x1, y1, x2, y2 = map(int, coords)  # Convert to integer coordinates
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
        cv2.putText(annotated_frame, 'Tire', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label

    # Apply contour detection to distinguish multiple tires within the same bounding box
    detect_contours(annotated_frame, tire_boxes)

    # Show the annotated frame
    cv2.imshow('Real-Time Tire Detection', annotated_frame)

    # Break the loop if the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
