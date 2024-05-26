# -*- coding: utf-8 -*-
import tensorflow as tf

# Set the path to the training images directory
train_images_path = r'path\to\train\images'

# Load the training images using TensorFlow's utility for image datasets,
# which automatically handles preprocessing like resizing and batching.
train_images = tf.keras.preprocessing.image_dataset_from_directory(
    train_images_path,
    image_size=(224, 224),  # Ensure the size matches the input size expected by the model.
    batch_size=32,  # Set the batch size for training.
)

# Define the path to the TensorFlow Lite model file.
tflite_model_path = "path\to\tflite\model\file.tflite"

# Open the TensorFlow Lite model file in binary read mode.
with open(tflite_model_path, 'rb') as file:
    tflite_model = file.read()

# Create a TFLiteConverter object from a saved TensorFlow model.
converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model)

# Apply uint8 quantization to optimize the model for size and speed.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.uint8]

# Define a representative data generator function that provides data samples
# from the training dataset to the converter for calibrating the quantization process.
def representative_data_gen():
    for input_value, _ in train_images.take(100):  # Use the first 100 batches as representative data.
        yield [input_value.numpy()]

# Assign the representative dataset function to the converter.
converter.representative_dataset = representative_data_gen

# Convert the model to a quantized TFLite model.
quantized_tflite_model = converter.convert()

# Specify the path where the quantized model will be saved.
output_model_path = 'path\to\save\quantized_model_uint8.tflite'

# Write the quantized model to a binary file.
with open(output_model_path, 'wb') as file:
    file.write(quantized_tflite_model)
