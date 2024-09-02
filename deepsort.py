from ultralytics import YOLO
import cv2
import numpy as np

# Define paths
model_path = r'yourpath\best.pt'
video_path = r'yourpath\test.'

# Load the YOLOv8 model trained to detect tires
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Video not found or cannot be opened: {video_path}")

# Initialize tire count and tracked objects
tire_count = 0
tracked_objects = {}
line_y = None
frame_index = 0

# Define text properties of tire count
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 4
text_color = (0, 0, 0)  # Black color

def put_text_bold(img, text, position, font, scale, color, thickness):
    # Draw the text multiple times with slight offsets to simulate a bold effect
    for offset in range(2):  # Adjust the range for more boldness
        x_offset = offset
        y_offset = offset
        cv2.putText(img, text, (position[0] + x_offset, position[1] + y_offset),
                    font, scale, color, thickness, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frames are returned

    frame_index += 1

    # Perform detection on the current frame
    results = model(frame)
    
    # Extract the bounding boxes from the results
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()     # Get confidence scores

    # Initialize new tracked objects for this frame
    current_objects = {}

    # Process each detected bounding box
    for bbox, score in zip(detections, scores):
        if score > 0.5:  # Adjust confidence threshold if necessary
            x1, y1, x2, y2 = map(int, bbox)
            bbox_tuple = (x1, y1, x2, y2)
            
            # Generate a unique ID based on distance to previously tracked objects
            obj_id = None
            min_dist = float('inf')

            for prev_id, prev_bbox in tracked_objects.items():
                prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
                dist = np.linalg.norm(np.array([x1, y1, x2, y2]) - np.array([prev_x1, prev_y1, prev_x2, prev_y2]))
                if dist < min_dist:
                    min_dist = dist
                    obj_id = prev_id
            
            if obj_id is None or min_dist > 50:  # 50 pixels threshold for new object
                obj_id = f"{frame_index}_{len(current_objects)}"

            # Add object to current tracked objects
            current_objects[obj_id] = bbox_tuple

            # Check if object has crossed the line and not yet counted
            if line_y is not None and obj_id in tracked_objects:
                prev_y1 = tracked_objects[obj_id][1]
                if prev_y1 < line_y <= y1:
                    tire_count += 1
                    print(f"Tire counted: {obj_id}, New count: {tire_count}")

    # Update tracked objects
    tracked_objects = current_objects

    # Draw a line for counting
    if line_y is None:
        line_y = frame.shape[0] // 2  # Define the middle of the frame as the line
    
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)  # Green line

    # Draw bounding boxes and tire count on the frame
    for bbox in tracked_objects.values():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw bounding box
        cv2.putText(frame, 'Tire', (x1, y1 - 10), font, 0.75, (255, 0, 0), 2)

    # Draw the tire count in black and bold
    put_text_bold(frame, f'Tire Count: {tire_count}', (10, 30), font, font_scale, text_color, font_thickness)

    # Show the annotated frame
    cv2.imshow('Tire Detection', frame)

    # Break the loop if the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
