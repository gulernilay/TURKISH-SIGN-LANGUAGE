#Define libraries
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt



# Define data directories
train_data_dir = 'E:\TurkishSignLanguage\Dataset\Main2Dataset_224\AugmentedTraining'
#test_data_dir='C:\\Users\\nilay\\Desktop\\Dataset\\MainDataset_224\\Test'

# Define image size and input shape
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
"""
test_generator = train_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')
"""

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(25, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# RAdam optimizer
# optimizer = tfa.optimizers.RectifiedAdam(learning_rate=0.001)

# AdamW optimizer with weight decay
#weight_decay = 1e-5
#optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=weight_decay)

# Define the learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# Use the learning rate schedule in the Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler function
#def scheduler(epoch, lr):
#    if epoch < 10:
#        return lr
#    else:
#        return lr * tf.math.exp(-0.1)

# Create callbacks for learning rate scheduling and early stopping
lr_scheduler = LearningRateScheduler(lr_schedule)
#early_stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with callbacks
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
   # callbacks=[lr_scheduler, early_stopper]
    callbacks=[lr_scheduler])

# Evaluate the model on the test data
#test_loss, test_accuracy = model.evaluate(test_generator)
#print(f"Test Accuracy: {test_accuracy}")

# Create the destination directory if it doesn't exist
destination_dir = 'C:\\Users\\nilay\\Desktop\\Dataset\\Main2Dataset_224\\Training'
os.makedirs(destination_dir, exist_ok=True)

# Save the TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(destination_dir, 'Dataset3_MobileNetV2.tflite'), 'wb') as f:
    f.write(tflite_model)

# Save the TensorFlow model
model.save(os.path.join(destination_dir, 'Dataset3_MobileNetV2.h5'))

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
