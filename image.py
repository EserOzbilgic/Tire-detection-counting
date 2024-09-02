from ultralytics import YOLO
import cv2

# Define paths
model_path = r'yourpath\best.pt'
image_path = r'yourpath\test.'

# Load the YOLOv8 model trained to detect tires
model = YOLO(model_path)

# Load the image
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Image not found or cannot be read: {img_path}")

# Perform detection
results = model(img)

# Filter out only tire detections based on class names
# Assuming your data.yaml specifies 'tire' as one of the classes
# If 'tire' is not the exact name in the classes, modify it accordingly
filtered_results = [result for result in results if result.boxes.cls[0] == 'tire']

# Plot and display the results for tire detection
annotated_img = results[0].plot()  # Access the first result and plot the bounding boxes

cv2.imshow('Tire Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()