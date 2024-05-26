import cv2
import numpy as np
import tensorflow as tf
import os


# TensorFlow Lite modelini yükleme
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\nilay\\Documents\\GitHub\\TURKISH-SIGN-LANGUAGE\\NewApprach_Training\\hand_model.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:\\Users\\nilay\\Documents\\GitHub\\TURKISH-SIGN-LANGUAGE\\Implementation of Neural Network\\HANDmodelBULUNDU.tflite") 
interpreter.allocate_tensors()

# Giriş ve çıkış detaylarını alın
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Giriş ve çıkış boyutlarını tanımlayın
input_height, input_width = input_shape[1], input_shape[2]
output_shape = output_details[0]['shape']
num_classes = output_shape[1]
num_boxes = output_shape[2]

# Paths for input and output
input_path = r"E:\\TurkishSignLanguage\\Dataset\\Main2Dataset_224\\Test"
output_path = r"E:\\TurkishSignLanguage\\Dataset\\Main2Dataset_224\\ProcessedTest"

folders_to_process = ['R', 'space']

def convert_bitmap_to_byte_buffer(bitmap):
    byte_buffer = np.zeros(4 * bitmap.shape[0] * bitmap.shape[1] * 3, dtype=np.uint8)
    int_values = bitmap.flatten()

    pixel = 0
    for val in int_values:
        byte_buffer[pixel * 3] = (((val >> 16) & 0xFF) / 255.0).astype(np.float32)
        byte_buffer[pixel * 3 + 1] = (((val >> 8) & 0xFF) / 255.0).astype(np.float32)
        byte_buffer[pixel * 3 + 2] = ((val & 0xFF) / 255.0).astype(np.float32)
        pixel += 1
    print("Byte buffer shape:", byte_buffer.shape)
    return byte_buffer

def recognize_image(mat_image, interpreter):
    # Resmi döndürme ve yansıtma işlemlerini gerçekleştirin
    rotated_mat_image = cv2.transpose(mat_image)
    rotated_mat_image = cv2.flip(rotated_mat_image, 1)

    # Bitmap'e dönüştürme
    height, width = rotated_mat_image.shape[:2]
    bitmap = cv2.cvtColor(rotated_mat_image, cv2.COLOR_BGR2RGB)

    # Giriş boyutunu modele uygun şekilde ölçekleme
    scaled_bitmap = cv2.resize(bitmap, (224, 224))

    # Bitmap'i byte buffer'a dönüştürme
    byte_buffer = convert_bitmap_to_byte_buffer(scaled_bitmap)

    # Resize image to match model input shape
    resized_image = cv2.resize(byte_buffer, (300, 300))

    # Add batch dimension to match model input shape
    input_data = np.expand_dims(resized_image, axis=0)

    # Convert input_data to FLOAT32
    input_data2 = input_data.astype(np.float32)
    input_data2 = np.expand_dims(input_data2, axis=-1)  # Add color channel as the last dimension
    input_data2 = np.repeat(input_data2, 3, axis=-1)  # Repeat color channel axis 3 times
    
    # Set input tensor for the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data2)
    interpreter.invoke()

    # Retrieving model outputs
    output_locations = interpreter.get_tensor(output_details[0]['index'])
    output_classes = interpreter.get_tensor(output_details[1]['index'])
    output_scores = interpreter.get_tensor(output_details[2]['index'])

    # Find the index of the box with the highest score
    max_score_index = np.argmax(output_scores[0])

    # Get the coordinates of the box with the highest score
    box = output_locations[0][max_score_index]
    y1 = int(box[0] * height)
    x1 = int(box[1] * width)
    y2 = int(box[2] * height)
    x2 = int(box[3] * width)

    # Draw the bounding box
    cv2.rectangle(rotated_mat_image, (x1,  y1), (x2, y2), (0, 255, 0), 2)

    # Crop an image to the region defined by a bounding box 
    # Crop an image to the region defined by a bounding box 
    if y1 < y2 and x1 < x2:
        cropped_region = rotated_mat_image[y1:y2, x1:x2]
    else:
        print("Invalid bounding box dimensions")
        return None
    cropped_rotated_region = cv2.transpose(cropped_region)
    cropped_rotated_region = cv2.flip(cropped_rotated_region, 0)


    return cropped_rotated_region

     #return mat_image

def process_image(filename, input_folder):
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {filename}")
        return

    processed_image = recognize_image(image, interpreter)
    if processed_image is None or processed_image.size == 0:
        print(f"Processed image is empty for {filename}")
        return

    output_image_path = os.path.join(output_path, filename)
    cv2.imwrite(output_image_path, processed_image)
    print(f"{filename} processed and saved at {output_image_path}.")

if __name__ == "__main__":
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for folder in folders_to_process:
        folder_path = os.path.join(input_path, folder)
        if os.path.exists(folder_path):
            print(f"Processing folder: {folder}")
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    process_image(filename, folder_path)
            print(f"Finished processing folder: {folder}")
        else:
            print(f"Folder {folder} does not exist")

    print("All specified folders have been processed.")


"""
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
        processed_image = recognize_image(image, interpreter)

        # Çıktı yolunu oluştur
        output_image_path = os.path.join(output_path, filename)

        # İşlenmiş görüntüyü kaydet
        cv2.imwrite(output_image_path, processed_image)

        print(f"{filename} işlendi ve kaydedildi.")

print("Tüm görüntüler işlendi ve kaydedildi.")
"""