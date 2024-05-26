import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# TensorFlow Lite interpreter oluştur ve modeli yükle
model_path = r"C:\Users\nilay\Desktop\Dataset\Main2Dataset_224\veriseti3_denemeler\deneme3_30epoch.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Giriş ve çıkış detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test veri seti için ImageDataGenerator tanımla ve generator oluştur
test_datagen = ImageDataGenerator(rescale=1./255)
test_data_dir = r'E:\TurkishSignLanguage\Dataset\Main2Dataset_224\mini_test'
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(input_details[0]['shape'][1], input_details[0]['shape'][2]),  # Modelin beklediği boyuta ayarla
    batch_size=1,  # Batch boyutunu 1 olarak ayarla
    class_mode='categorical',
    shuffle=False)

# Modeli test veri seti üzerinde değerlendir
total_accuracy = 0
count = 0
for images, labels in test_generator:
    # Giriş tensorunu ayarla
    interpreter.set_tensor(input_details[0]['index'], images.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Çıktıları değerlendir
    predicted_labels = np.argmax(output_data, axis=1)
    true_labels = np.argmax(labels, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    total_accuracy += accuracy
    count += 1
    if count >= len(test_generator.labels):
        break

overall_accuracy = total_accuracy / count
print(f"Test accuracy: {overall_accuracy}")
