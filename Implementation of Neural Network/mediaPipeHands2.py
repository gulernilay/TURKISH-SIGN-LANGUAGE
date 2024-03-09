# resimden koordinatları çıkartmak , ve koordinatlı resimleri kaydetme , koordinatları ekstra olarak kaydetme 


import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Load an image
image = cv2.imread("E:\TurkishSignLanguage\TRAINING\originalimages\A\A_0_0.jpg")  # Update with the path to your image file

# Convert the image from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and extract hand landmarks
results = hands.process(image)

# Check if landmarks were found and then write them to a CSV file with the label
if results.multi_hand_landmarks:
    with open('hand_landmarks.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for hand_landmarks in results.multi_hand_landmarks:
            # Write landmarks to CSV along with the label for the sign (e.g., 'A')
            row = ['A']  # Start with the label
            for lm in hand_landmarks.landmark:
                # Add x, y, and z coordinates to the row
                row.extend([lm.x, lm.y, lm.z])
            writer.writerow(row)

# Display the image with drawn landmarks (optional)
# Note: To display the image with landmarks drawn, convert it back to BGR
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
cv2.imshow('MediaPipe Hands', image)
cv2.waitKey(0)  # Wait for a key press to close the image window

# Clean up
cv2.destroyAllWindows()
hands.close()
