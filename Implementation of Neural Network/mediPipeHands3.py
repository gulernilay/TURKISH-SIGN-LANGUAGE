import cv2
import mediapipe as mp
import csv
import os
# R-S-T-U-V-Y-Z 
# Function to process an image and extract hand landmarks
def process_image(image_path, label,csv_writer,destination_dir):
    # Load an image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and extract hand landmarks
    results = hands.process(image_rgb)

    # Convert the processed image back from RGB to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Check if landmarks were found and write them to the CSV file with the label
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp.solutions.drawing_utils.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Write landmarks to CSV along with the label for the sign
            row = [label]
            for lm in hand_landmarks.landmark:
                # Add x, y, and z coordinates to the row
                row.extend([lm.x, lm.y, lm.z])
            csv_writer.writerow(row)
            
            # Define the save path for the marked image
            base_name = os.path.basename(image_path)
            save_path = os.path.join(destination_dir, base_name)
            
            # Save the image with drawn landmarks
            cv2.imwrite(save_path, image_bgr)
# Define the destination directory for images
destination_dir = "E:\\TÜRK İŞARET DİLİ\\Bosphorus\\archive.zip\\tr_signLanguage_dataset\\test\A"


# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Path to the directory containing images
image_dir = "E:\\TurkishSignLanguage\\TRAINING\\Test\\A"


# Open CSV file to write landmarks data
with open('hand_landmarks_test_A.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Loop over each image in the directory
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(image_dir, image_name)
            process_image(image_path, 'A', writer, destination_dir)  # The label 'A' is used for all images in this folder

# Clean up
hands.close()
