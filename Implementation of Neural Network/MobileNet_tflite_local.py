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


# Define the destination directory
destination_dir = 'E:\TurkishSignLanguage\CoordinatedImages\Images'

# Kaggle API'ye erişim için kimlik bilgilerini ayarla
#os.environ['KAGGLE_USERNAME'] = "nilaygler"
#os.environ['KAGGLE_KEY'] = "da17a3301c57c4360bf0a34c9befec20"

#import subprocess

# Command to download the dataset
#command = 'kaggle datasets download -d berkaykocaoglu/tr-sign-language'

# Run the command
#subprocess.run(command, shell=True)


# İndirilen zip dosyasını açma
#with zipfile.ZipFile('tr-sign-language.zip', 'r') as zip_ref:
#    zip_ref.extractall('/content/dataset')                 


# Define data directories
train_data_dir = 'E:\TurkishSignLanguage\CoordinatedImages\Images\Training'
test_data_dir = 'E:\TurkishSignLanguage\CoordinatedImages\Images\Test'
#validation_data_dir="E:\\MELANOMA\\FENG497-MELANOMA\\5binaddet224\\validation"

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
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

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
history = model.fit(train_generator,    # back propogation 
                    epochs=25,
                    validation_data=validation_generator)  # Validation data ekleyin

# Test accuracy hesaplayın
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")

# %%
# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)
# Save the TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(destination_dir, 'Coordinated_mobileNet_model.tflite'), 'wb') as f:
    f.write(tflite_model)

# Save the TensorFlow model
model.save(os.path.join(destination_dir, 'Coordinated_mobileNet_model.h5'))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()