# Focal Loss , Adamw optimizer and Early Stopping is applied

import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Disable oneDNN optimizations to avoid potential incompatibility issues or bugs.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up the paths for the dataset.
destination_dir = 'path\to\destination\dir'
train_data_dir = 'path\to\train\data\dir'

# Define the image size and input shape for the model.
image_size = (224, 224)
input_shape = image_size + (3,)

# Set the batch size for training and validation.
batch_size = 16

# Configure data generators with data augmentation and validation split.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  

# Prepare the training data generator.
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

# Prepare the validation data generator.
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Define Focal Loss function for model training.
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = -alpha * (y_true * tf.math.pow(1. - y_pred, gamma) * tf.math.log(y_pred)) - ((1. - alpha) * tf.math.pow(y_pred, gamma) * tf.math.log(1. - y_pred))
    return tf.reduce_mean(loss)

# Load the MobileNet model pre-trained on ImageNet without the top layer.
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Extend the base model.
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

# Create the final model.
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model.
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with AdamW optimizer and focal loss.
optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-5)
model.compile(optimizer=optimizer, loss=focal_loss, metrics=['accuracy'])

# Configure early stopping.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with the training and validation data.
history = model.fit(train_generator, epochs=1, validation_data=validation_generator, callbacks=[early_stopping])

# Ensure the destination directory exists.
os.makedirs(destination_dir, exist_ok=True)

# Convert the trained model to TensorFlow Lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True  # Enable detailed logging with the new converter.
tflite_model = converter.convert()

# Save the TensorFlow Lite model.
tflite_model_path = "converted_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model converted successfully and saved to {tflite_model_path}")

# Plot training and validation accuracy.
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot training and validation loss.
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
