import cv2
import os
import numpy as np

# Define paths relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "./lego_ref_images/og")
output_dir = os.path.join(script_dir, "./lego_ref_images/og_cropped")
model_dir = os.path.join(script_dir, "models")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load Deep Learning Face Detector (ResNet SSD)
prototxt_path = os.path.join(model_dir, "deploy.prototxt")
model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    raise FileNotFoundError("Face detection model files not found in 'models' directory!")

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

face_count = 0  # Counter for detected and cropped faces
total_images = len([f for f in os.listdir(input_dir) if f.endswith(".png")])  # Count total PNG images

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):  # Only process PNG files
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Convert image to blob for DNN processing
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

        # Perform face detection
        face_net.setInput(blob)
        detections = face_net.forward()

        max_confidence = 0
        best_face = None

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Get confidence score

            if confidence > 0.5:  # Only consider confident detections
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")

                if confidence > max_confidence:
                    max_confidence = confidence
                    best_face = (x, y, x2 - x, y2 - y)  # Store best face

        # If at least one face is detected, crop the best one
        if best_face:
            x, y, w, h = best_face

            # Expand the bounding box to include more of the head and hairstyle
            expand_ratio = 0.4  # Increase the bounding box by 40%
            x = max(0, x - int(w * expand_ratio / 2))
            y = max(0, y - int(h * expand_ratio / 2))
            w = min(w + int(w * expand_ratio), img.shape[1] - x)
            h = min(h + int(h * expand_ratio), img.shape[0] - y)
            
            cropped_face = img[y:y+h, x:x+w]  # Crop expanded face region

            # Save cropped image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cropped_face)
            face_count += 1
            print(f"Cropped and saved: {output_path}")
        else:
            print(f"No face detected in: {filename}")

print(f"Processing complete! {face_count} faces detected and cropped out of {total_images} total images.")