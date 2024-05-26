

from importlib.metadata import files
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import matplotlib.pyplot as pltmport
import shutil
import os
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout ,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Disable oneDNN optimizations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

train_data_dir = r'E:\TurkishSignLanguage\Dataset\Main2Dataset_224\AugmentedTraining'

# Define image size
image_size = (224, 224)
input_shape = image_size + (3,)
# Define batch size
batch_size = 16

# Define data generators
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
    subset='training')  # training verileri)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # validation verileri

# Load pre-trained MobileNet model without the top layer
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Add new layers on top of MobileNet
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # L2 regularization
x = BatchNormalization()(x)
x = Dropout(0.3)(x)  # Dropout layer
predictions = Dense(25, activation='softmax')(x)  # Output layer with softmax for multi-class classification

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# AdamW optimizatörünü kullanın
optimizer = tf.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.0001)

# Compile the model with a built-in loss function
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Assuming train_generator and validation_generator are defined as per your dataset
# Train the model
history = model.fit(train_generator, epochs=25, validation_data=validation_generator, callbacks=[early_stopping])

# First, save your Keras model to the Kaggle working directory
model_path = r'C:\Users\nilay\Desktop\Dataset\Main2Dataset_224\veriseti3_denemeler\model2.keras'
model.save(model_path)
print(f"Keras model saved to {model_path}")


# Load the model without compiling it
model = load_model(model_path, compile=False)

# Assuming your model is defined/loaded here, if not already loaded
# model = load_model('path_to_your_model')

print("Starting model compilation...")
# Manually compile the model with the standard loss
model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model compiled successfully.")
print("Starting model conversion to TensorFlow Lite...")

import tensorflow as tf
# Convert the model to TensorFlow Lite
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Model conversion successful.")
except Exception as e:
    print(f"Failed to convert model: {e}")

# Save the converted model
tflite_model_path = r'C:\Users\nilay\Desktop\Dataset\Main2Dataset_224\veriseti3_denemeler\model2.tflite'
print(f"Saving the model to {tflite_model_path}...")
try:
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model saved successfully to {tflite_model_path}")
except Exception as e:
    print(f"Failed to save the model: {e}")


#deneme 2 kodu 



