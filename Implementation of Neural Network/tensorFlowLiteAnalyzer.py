import tensorflow as tf
import os
import numpy as np

# Model dosyasının yolu
model_path = r"C:\Users\nilay\Downloads\detect (2).tflite"

# TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path=model_path)

# Modeli ayarla
interpreter.allocate_tensors()

# Model bilgilerini al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Giriş ve çıkış tensorlerini yazdır
print('Input Tensor Details:')
for input_detail in input_details:
    print(input_detail)

print('\nOutput Tensor Details:')
for output_detail in output_details:
    print(output_detail)

# Model boyutunu hesapla
model_size = os.path.getsize(model_path)
print('\nModel Size: {:.2f} KB'.format(model_size / 1024))

# İşlem sayısını hesapla (ops sayısı)
total_ops = 0
for layer in interpreter.get_tensor_details():
    # Her katmanın işlem sayısını hesapla ve topla
    if 'shape' in layer:
        total_ops += np.prod(layer['shape'])
print('Total Operations: {}'.format(total_ops))
