import cv2
import numpy as np
import tensorflow as tf
import os

# TensorFlow Lite modelini yükleme
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\nilay\\Documents\\GitHub\\TURKISH-SIGN-LANGUAGE\\NewApprach_Training\\hand_model.tflite")
interpreter.allocate_tensors()

# Model giriş ve çıkış detayları
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Giriş ve çıkış boyutları
input_height, input_width = input_shape[1], input_shape[2]
output_shape = output_details[0]['shape']
num_classes = output_shape[1]
num_boxes = output_shape[2]

# Veri yolu ve çıktı yolu
input_path = f"E:\TurkishSignLanguage\Dataset\Main2Dataset_224\Training\C"
output_path = f"E:\TurkishSignLanguage\Dataset\Main2Dataset_224\ProcessedTraining\C"


# İşlenmiş görüntüyü geri döndüren fonksiyon
def recogImage(mat_image):
    # Resmi döndürme ve yansıtma işlemlerini gerçekleştirin
    rotated_mat_image = cv2.transpose(mat_image)
    rotated_mat_image = cv2.flip(rotated_mat_image, 1)

    # Resmi boyutlandırın
    scaled_image = cv2.resize(rotated_mat_image, (input_width, input_height))

    # Resmi modele uygun formatta hazırlayın
    input_data = np.expand_dims(scaled_image, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    # Giriş verisini modele yükleme
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Modeli çalıştırma
    interpreter.invoke()

    # Çıktıları alma
    output_locations = interpreter.get_tensor(output_details[0]['index'])
    output_classes = interpreter.get_tensor(output_details[1]['index'])
    output_scores = interpreter.get_tensor(output_details[2]['index'])

    # Kutuları çizme ve etiketleme işlemlerini gerçekleştirin
    for i in range(num_boxes):
        if output_scores[0, i] > 0.5:
            y1, x1, y2, x2 = output_locations[0, i]
            y1, x1, y2, x2 = int(y1 * input_height), int(x1 * input_width), int(y2 * input_height), int(x2 * input_width)
            cv2.rectangle(rotated_mat_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Kamerayı geri döndürün ve -90 derece çevirin
    mat_image = cv2.transpose(rotated_mat_image)
    mat_image = cv2.flip(mat_image, 0)
   

    return mat_image

# Eğer çıktı dizini yoksa oluştur
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Tüm görüntü dosyalarını işle
for filename in os.listdir(input_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Görüntüyü oku
        image_path = os.path.join(input_path, filename)
        image = cv2.imread(image_path)

        # Görüntüyü işle
        processed_image = recogImage(image)

        # Çıktı yolunu oluştur
        output_image_path = os.path.join(output_path, filename)

        # İşlenmiş görüntüyü kaydet
        cv2.imwrite(output_image_path, processed_image)

        print(f"{filename} işlendi ve kaydedildi.")

print("Tüm görüntüler işlendi ve kaydedildi.")


