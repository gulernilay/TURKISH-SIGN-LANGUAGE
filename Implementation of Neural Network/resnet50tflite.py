import os
import zipfile
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt


# Kaggle API'ye erişim için kimlik bilgilerini ayarla
os.environ['KAGGLE_USERNAME'] = "nilaygler"
os.environ['KAGGLE_KEY'] = "da17a3301c57c4360bf0a34c9befec20"

import subprocess

# Command to download the dataset
command = 'kaggle datasets download -d berkaykocaoglu/tr-sign-language'

# Run the command
subprocess.run(command, shell=True)


# İndirilen zip dosyasını açma
with zipfile.ZipFile('tr-sign-language.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')                 


# Define data directories
train_data_dir = '/content/dataset/tr_signLanguage_dataset/train'
test_data_dir = '/content/dataset/tr_signLanguage_dataset/test'

# Define image size
image_size = (224, 224)
input_shape = image_size + (3,)

# Define batch size
batch_size = 32

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3)  # %30'unu validation olarak kullan)

test_datagen = ImageDataGenerator(rescale=1./255)

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

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers for our classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(26, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=20,
                    validation_data=validation_generator)  # Validation data ekleyin

# Test accuracy hesaplayın
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")

# Save the TensorFlow model
model.save('resnet50_model.h5')

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('resnet50_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()       