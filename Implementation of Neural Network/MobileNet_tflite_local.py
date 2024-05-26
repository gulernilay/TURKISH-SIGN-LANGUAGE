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
destination_dir = 'C:\\Users\\nilay\\Desktop\\Dataset\\Main2Dataset_224'          
# Define data directories
train_data_dir = 'E:\\TurkishSignLanguage\\Dataset\\Main2Dataset_224\\AugmentedTraining'
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
"""

#test_data_dir = 'C:\\Users\\nilay\\Desktop\\Dataset\\MainDataset_224\\Test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

    # Test accuracy hesaplayın
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")


class FocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate focal loss
        focal_loss = self.alpha * tf.pow(1 - y_pred, self.gamma) * cross_entropy

        # Sum over classes
        return tf.reduce_sum(focal_loss, axis=-1)
"""
# Define Focal Loss function
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    # Calculate focal loss
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    focal_loss = -alpha * (y_true * tf.math.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)) - ((1.0 - alpha) * tf.math.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(focal_loss)

# Use AdamW optimizer with weight decay
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-5)

# Load pre-trained ResNet50 model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers for our classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
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


# Train the model
history = model.fit(train_generator,    # back propogation 
                    epochs=35,
                    validation_data=validation_generator)  # Validation data ekleyin

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)
# Save the TensorFlow Lite model
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#with open(os.path.join(destination_dir, 'MobileNet_ownDataset3.tflite'), 'wb') as f:
    #f.write(tflite_model)

# Save the TensorFlow model
model.save(os.path.join(destination_dir, 'MobileNet_ownDataset3.h5'))

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

"""
#adamw , focal loss ile : 
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 746s 422ms/step - accuracy: 0.6649 - loss: 0.0135 - val_accuracy: 0.8205 - val_loss: 0.0095
Epoch 2/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 804s 457ms/step - accuracy: 0.8833 - loss: 0.0080 - val_accuracy: 0.8322 - val_loss: 0.0091
Epoch 3/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 809s 460ms/step - accuracy: 0.9159 - loss: 0.0074 - val_accuracy: 0.8463 - val_loss: 0.0089
Epoch 4/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 828s 471ms/step - accuracy: 0.9285 - loss: 0.0072 - val_accuracy: 0.8651 - val_loss: 0.0084
Epoch 5/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 812s 462ms/step - accuracy: 0.9363 - loss: 0.0070 - val_accuracy: 0.8677 - val_loss: 0.0084
Epoch 6/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 861s 489ms/step - accuracy: 0.9401 - loss: 0.0069 - val_accuracy: 0.8754 - val_loss: 0.0082
Epoch 7/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 782s 445ms/step - accuracy: 0.9451 - loss: 0.0068 - val_accuracy: 0.8637 - val_loss: 0.0083
Epoch 8/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 776s 441ms/step - accuracy: 0.9488 - loss: 0.0067 - val_accuracy: 0.8728 - val_loss: 0.0080
Epoch 9/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 721s 410ms/step - accuracy: 0.9467 - loss: 0.0066 - val_accuracy: 0.8717 - val_loss: 0.0081
Epoch 10/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 729s 414ms/step - accuracy: 0.9523 - loss: 0.0066 - val_accuracy: 0.8799 - val_loss: 0.0080
Epoch 11/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 862s 490ms/step - accuracy: 0.9536 - loss: 0.0065 - val_accuracy: 0.8808 - val_loss: 0.0079
Epoch 12/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 831s 472ms/step - accuracy: 0.9523 - loss: 0.0065 - val_accuracy: 0.8772 - val_loss: 0.0079
Epoch 13/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 814s 463ms/step - accuracy: 0.9533 - loss: 0.0064 - val_accuracy: 0.8767 - val_loss: 0.0079
Epoch 14/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 795s 452ms/step - accuracy: 0.9546 - loss: 0.0064 - val_accuracy: 0.8836 - val_loss: 0.0080
Epoch 15/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 778s 442ms/step - accuracy: 0.9531 - loss: 0.0064 - val_accuracy: 0.8816 - val_loss: 0.0079
Epoch 16/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 818s 465ms/step - accuracy: 0.9554 - loss: 0.0064 - val_accuracy: 0.8818 - val_loss: 0.0079
Epoch 17/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 750s 426ms/step - accuracy: 0.9561 - loss: 0.0064 - val_accuracy: 0.8805 - val_loss: 0.0078
Epoch 18/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 732s 416ms/step - accuracy: 0.9554 - loss: 0.0063 - val_accuracy: 0.8781 - val_loss: 0.0079
Epoch 19/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 841s 478ms/step - accuracy: 0.9593 - loss: 0.0063 - val_accuracy: 0.8792 - val_loss: 0.0079
Epoch 20/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 747s 425ms/step - accuracy: 0.9554 - loss: 0.0063 - val_accuracy: 0.8834 - val_loss: 0.0079
Epoch 21/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 760s 432ms/step - accuracy: 0.9548 - loss: 0.0063 - val_accuracy: 0.8846 - val_loss: 0.0078
Epoch 22/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 768s 436ms/step - accuracy: 0.9560 - loss: 0.0063 - val_accuracy: 0.8797 - val_loss: 0.0077
Epoch 23/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 819s 466ms/step - accuracy: 0.9590 - loss: 0.0063 - val_accuracy: 0.8824 - val_loss: 0.0077
Epoch 24/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 757s 430ms/step - accuracy: 0.9597 - loss: 0.0062 - val_accuracy: 0.8834 - val_loss: 0.0077
Epoch 25/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 753s 428ms/step - accuracy: 0.9569 - loss: 0.0062 - val_accuracy: 0.8814 - val_loss: 0.0078
Epoch 26/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 770s 438ms/step - accuracy: 0.9591 - loss: 0.0062 - val_accuracy: 0.8805 - val_loss: 0.0078
Epoch 27/35
1757/1757 ━━━━━━━━━━━━━━━━━━━━ 1016s 578ms/step - accuracy: 0.9585 - loss: 0.0062 - val_accuracy: 0.8818 - val_loss: 0.0077




"""
#adamw , focal loss ile : 
#Epoch 11/35
#1757/1757 ━━━━━━━━━━━━━━━━━━━━ 862s 490ms/step - accuracy: 0.9536 - loss: 0.0065 - val_accuracy: 0.8808 - val_loss: 0.0079
#Epoch 12/35
#1757/1757 ━━━━━━━━━━━━━━━━━━━━ 831s 472ms/step - accuracy: 0.9523 - loss: 0.0065 - val_accuracy: 0.8772 - val_loss: 0.0079
#Epoch 13/35
#1757/1757 ━━━━━━━━━━━━━━━━━━━━ 814s 463ms/step - accuracy: 0.9533 - loss: 0.0064 - val_accuracy: 0.8767 - val_loss: 0.0079
#Epoch 14/35
#1757/1757 ━━━━━━━━━━━━━━━━━━━━ 795s 452ms/step - accuracy: 0.9546 - loss: 0.0064 - val_accuracy: 0.8836 - val_loss: 0.0080

