
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
import matplotlib.pyplot as plt
import shutil
import os
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay

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

lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=100000, decay_rate=0.96, staircase=True)


# Load pre-trained ResNet50 model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Add dropout layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(dropout_rate)(x)  # Add dropout layer
predictions = Dense(25, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Use AdamW optimizer with weight decay
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5)

# Compile the model
model.compile(optimizer=optimizer, loss=focal_loss, metrics=['accuracy'])

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_generator, epochs=1, validation_data=validation_generator, callbacks=[early_stopping])

os.makedirs(destination_dir, exist_ok=True)
# Save the TensorFlow model
# Convert the model to TFLite
# Save the TensorFlow model

# Save the TensorFlow model
model.save('resnet50_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = os.path.join(destination_dir, 'MobileNet_ownDataset3_dropout.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("TFLite modeli başarıyla kaydedildi:", tflite_model_path)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


# adamw , dropfactor , learning rate scheduler , focalloss implementation kodudurç 