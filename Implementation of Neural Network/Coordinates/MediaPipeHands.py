import cv2
import mediapipe as mp
import csv

# MediaPipe Hands modelini başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Kamera görüntüsünü oku
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Kamera görüntüsü alınamıyor.")
        continue

    # Görüntüyü BGR'den RGB'ye dönüştür
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # El algılama işlemi
    results = hands.process(image)

    # Algılanan ellerin anahtar noktalarını çiz
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Her bir anahtar noktanın x, y ve z koordinatlarını al ve kaydet
            hand_landmarks_list = []
            for lm in hand_landmarks.landmark:
                hand_landmarks_list.append([lm.x, lm.y, lm.z])
            
            # Koordinatları CSV dosyasına yaz
            with open('hand_landmarks.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(hand_landmarks_list)

    # Görüntüyü göster
    cv2.imshow('MediaPipe Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
