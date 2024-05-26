# adamw optimizer , dropfactor(0.3)  , learning rate scheduler and  focal loss is implemented.

import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Define directory paths
destination_dir = 'path\\to\\destination\\dir'
train_data_dir = 'path\\to\\train\\data\\dir'

# Define image processing parameters
image_size = (224, 224)
input_shape = image_size + (3,)
batch_size = 16

# Setup data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Focal loss function for handling class imbalance
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = -alpha * (y_true * tf.math.pow(1. - y_pred, gamma) * tf.math.log(y_pred)) - ((1. - alpha) * tf.math.pow(y_pred, gamma) * tf.math.log(1. - y_pred))
    return tf.reduce_mean(loss)

# Learning rate scheduler
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# Load and modify the MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)  # Apply dropout to reduce overfitting
predictions = Dense(25, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the convolutional base to reuse prelearned features
for layer in base_model.layers:
    layer.trainable = False

# Setup the model for training
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
model.compile(optimizer=optimizer, loss=focal_loss, metrics=['accuracy'])

# Early stopping to halt training when validation loss is not improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=1,  # Set higher for real training
    validation_data=validation_generator,
    callbacks=[early_stopping])

# Ensure the directory exists and save the model
os.makedirs(destination_dir, exist_ok=True)
model.save('path\\to\\model\\resnet50_model.keras')

# Convert the trained model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = os.path.join(destination_dir, 'MobileNet_ownDataset3_dropout.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved successfully:", tflite_model_path)

# Plot the training and validation accuracy and loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
