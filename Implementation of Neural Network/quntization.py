# -*- coding: utf-8 -*-

import tensorflow as tf

# Eğitim verilerinin yolu
train_images_path = r'E:\TurkishSignLanguage\TRAINING\originalimages'


# Eğitim verilerini yükle
train_images = tf.keras.preprocessing.image_dataset_from_directory(
    train_images_path,
    image_size=(224, 224),  # Modelinizin giriş boyutuna uygun şekilde ayarlayın
    batch_size=32,  # Batch boyutunu belirleyin
)

# TensorFlow Lite modelini yükle
tflite_model_path = "C:\\Users\\nilay\\Desktop\\SignLanguage\\TURKISH-SIGN-LANGUAGE\\Implementation of Neural Network\\MobileNet.tflite"
with open(tflite_model_path, 'rb') as f:
    tflite_model = f.read()

# TFLiteConverter ile modeli tekrar dönüştür
converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model)

# uint8 quantization uygula
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.uint8]

# Eğitim verilerini temsil eden bir fonksiyon tanımla
def representative_data_gen():
    for input_value, _ in train_images.take(100):  # İlk 100 batch'i al
        yield [input_value.numpy()]

# Eğitim verilerini temsil eden fonksiyonu belirt
converter.representative_dataset = representative_data_gen

# Dönüştürülmüş modeli tekrar TFLite formatında kaydet
quantized_tflite_model = converter.convert()
with open('path/to/save/quantized_model_uint8.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
