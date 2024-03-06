# -*- coding: utf-8 -*-
import os

# TensorFlow'un dosya işlemleri için kullanacağı kodlamayı ayarlayın
os.environ['TF_ENABLE_FILESYSTEM_ENCODING'] = 'utf-8'

import tensorflow as tf
import numpy as np

# Eğitim verilerinizin bulunduğu klasör yolu
train_images_path = 'E:\\TurkishSignLanguage\\TRAINING\\originalimages'

# Eğitim verilerini bir TensorFlow dataset olarak yükle
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_images_path,
    image_size=(224, 224),  # Modelinizin giriş boyutuna uygun olarak ayarlayın
    batch_size=32,  # Batch boyutunu belirleyin
    label_mode=None  # Sadece resimleri yüklemek için etiket modunu None olarak ayarlayın
)
saved_model_dir = "C:\\Users\\nilay\\Desktop\\MobileNet.tflite"

def representative_data_gen():
    for input_value in train_dataset.take(100):  # İlk 100 batch'i al
        yield [input_value.numpy()]


# TFLiteConverter nesnesini oluşturun
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Optimizasyonları ve representative dataset fonksiyonunu ayarlayın
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Target spec ayarlarını belirtin
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.uint8]
converter.inference_input_type = tf.uint8  # Opsiyonel
converter.inference_output_type = tf.uint8  # Opsiyonel

# Modeli dönüştürün ve kaydedin
tflite_quant_model = converter.convert()
with open('quantized_model_uint8.tflite', 'wb') as f:
    f.write(tflite_quant_model)
