import cv2
import numpy as np
import os
from glob import glob

# Load the dataset
image_paths = glob('C:/Users/ASUS/PycharmProjects/face_recognition/part1/*.jpg')



# Function to extract ocular region
def extract_ocular_region(image):
    # Using Haar Cascades to detect eyes
    eye_cascade = cv2.CascadeClassifier('C:/Users/ASUS/PycharmProjects/face_recognition/cascade classifier/haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) >= 2:
        # Get coordinates for the first two eyes detected
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]
        # Crop region containing both eyes
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)
        ocular_region = image[y_min:y_max, x_min:x_max]
        #print(ocular_region)
        print('yes')
        return ocular_region
    return None


images = []
labels = []

for path in image_paths:
    try:
        # Extract age from the filename
        age = int(os.path.basename(path).split('_')[0])
        # Define age groups (example: 0-18, 19-35, 36-50, 51+)
        if age <= 18:
            label = '0-18'
        elif age <= 35:
            label = '19-35'
        elif age <= 50:
            label = '36-50'
        else:
            label = '51+'

        # Read the image
        image = cv2.imread(path)
        ocular_region = extract_ocular_region(image)
        if ocular_region is not None:
            images.append(cv2.resize(ocular_region, (64, 64)))

            labels.append(label)
    except Exception as e:
        print(f'Error processing image {path}: {e}')

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Save the preprocessed dataset
np.save('C:/Users/ASUS/PycharmProjects/face_recognition/ocular_images.npy', images)
print('Done with occular')
np.save('C:/Users/ASUS/PycharmProjects/face_recognition/age_labels.npy', labels)
print('Done with age')
