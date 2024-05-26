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

# Define the destination directory
destination_dir = 'C:\\Users\\nilay\\Desktop\\Dataset\\Main2Dataset_224'          
# Define data directories
train_data_dir = 'E:\\TurkishSignLanguage\\Dataset\\Main2Dataset_224\\AugmentedTraining'
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

# Define Focal Loss function
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    # Calculate focal loss
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    focal_loss = -alpha * (y_true * tf.math.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)) - ((1.0 - alpha) * tf.math.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(focal_loss)


# Define dropout rate
dropout_rate = 0.3

# Load pre-trained ResNet50 model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Add dropout layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  
#x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # Applying L2 regularization
x = BatchNormalization()(x)  # Adding Batch Normalization
x = Dropout(dropout_rate)(x)  # Add dropout layer
predictions = Dense(25, activation='softmax')(x)


# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Use AdamW optimizer with weight decay
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-5)

# Compile the model
model.compile(optimizer=optimizer, loss=focal_loss, metrics=['accuracy'])

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_generator, epochs=22, validation_data=validation_generator, callbacks=[early_stopping])


# First, save your Keras model to the Kaggle working directory
model_path = r'C:\Users\nilay\Desktop\Dataset\Main2Dataset_224\veriseti3_denemeler\deneme5.keras'
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
tflite_model_path = r'C:\Users\nilay\Desktop\Dataset\Main2Dataset_224\veriseti3_denemeler\deneme5.tflite'
print(f"Saving the model to {tflite_model_path}...")
try:
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model saved successfully to {tflite_model_path}")
except Exception as e:
    print(f"Failed to save the model: {e}")




"""
drop factor: 0.5  + focal loss + adamw optimizer
Epoch 1/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 696s 390ms/step - accuracy: 0.5031 - loss: 0.0170 - val_accuracy: 0.8158 - val_loss: 0.0097
Epoch 2/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 663s 377ms/step - accuracy: 0.8146 - loss: 0.0094 - val_accuracy: 0.8441 - val_loss: 0.0091
Epoch 3/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 665s 378ms/step - accuracy: 0.8564 - loss: 0.0085 - val_accuracy: 0.8556 - val_loss: 0.0087
Epoch 4/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 661s 376ms/step - accuracy: 0.8841 - loss: 0.0081 - val_accuracy: 0.8597 - val_loss: 0.0086
Epoch 5/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 666s 379ms/step - accuracy: 0.8973 - loss: 0.0078 - val_accuracy: 0.8662 - val_loss: 0.0084
Epoch 6/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 758s 431ms/step - accuracy: 0.9044 - loss: 0.0076 - val_accuracy: 0.8670 - val_loss: 0.0084
Epoch 7/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 835s 475ms/step - accuracy: 0.9118 - loss: 0.0074 - val_accuracy: 0.8583 - val_loss: 0.0085
Epoch 8/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 720s 409ms/step - accuracy: 0.9160 - loss: 0.0074 - val_accuracy: 0.8550 - val_loss: 0.0083
Epoch 9/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 755s 429ms/step - accuracy: 0.9200 - loss: 0.0073 - val_accuracy: 0.8701 - val_loss: 0.0083
Epoch 10/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 714s 405ms/step - accuracy: 0.9243 - loss: 0.0072 - val_accuracy: 0.8684 - val_loss: 0.0083
Epoch 11/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 737s 418ms/step - accuracy: 0.9256 - loss: 0.0071 - val_accuracy: 0.8697 - val_loss: 0.0080
Epoch 12/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 749s 426ms/step - accuracy: 0.9275 - loss: 0.0071 - val_accuracy: 0.8755 - val_loss: 0.0081
Epoch 13/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 740s 421ms/step - accuracy: 0.9288 - loss: 0.0070 - val_accuracy: 0.8662 - val_loss: 0.0081
Epoch 14/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 744s 423ms/step - accuracy: 0.9335 - loss: 0.0070 - val_accuracy: 0.8742 - val_loss: 0.0081
Epoch 15/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 752s 428ms/step - accuracy: 0.9351 - loss: 0.0070 - val_accuracy: 0.8725 - val_loss: 0.0081
Epoch 16/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 719s 409ms/step - accuracy: 0.9386 - loss: 0.0069 - val_accuracy: 0.8621 - val_loss: 0.0082



"""