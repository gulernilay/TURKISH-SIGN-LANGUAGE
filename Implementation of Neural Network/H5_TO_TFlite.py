
import tensorflow as tf
from tensorflow.keras.models import load_model

# Path to your saved .h5 model
h5_model_path = 'C:\\Users\\nilay\\Desktop\\Dataset\\Main2Dataset_224\\veriseti3_denemeler\\deneme2_dropout=0.5\\MobileNet_ownDataset3_dropout.h5'

# Define the custom focal loss function
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    fl = -alpha * (y_true * tf.math.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)) - ((1.0 - alpha) * (1.0 - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(fl)

# Custom objects need to be specified during the model load if your model uses any
custom_objects = {'focal_loss': lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.25, gamma=2)}

# Load the model with custom loss function
model = load_model(h5_model_path, custom_objects=custom_objects)
# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a file
tflite_model_path = 'C:\\Users\\nilay\\Desktop\\Dataset\\Main2Dataset_224\\MobileNet_ownDataset3_dropout.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite and saved at:", tflite_model_path)
