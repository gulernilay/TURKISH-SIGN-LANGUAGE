import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Custom Focal Loss implementation
#class FocalLoss(tf.keras.losses.Loss):
#    def __init__(self, gamma=2., alpha=4., reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='focal_loss'):
#        super(FocalLoss, self).__init__(reduction=reduction, name=name)
#        self.gamma = gamma
#        self.alpha = alpha

 #   def call(self, y_true, y_pred):
 #       epsilon = tf.keras.backend.epsilon()
 #       y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
 #       y_true = tf.cast(y_true, tf.float32)
 #       
 #       alpha_tensor = tf.multiply(y_true, self.alpha)
 #       cross_entropy = -y_true * tf.math.log(y_pred)
  #      loss = alpha_tensor * tf.pow(1 - y_pred, self.gamma) * cross_entropy
   #     return tf.reduce_mean(loss)


# Define paths to the datasets
train_dir = "E:\MELANOMA\FENG497-MELANOMA\APPROACH2\MODEL6_VASC\\5k\\training"
val_dir = "E:\MELANOMA\FENG497-MELANOMA\APPROACH2\MODEL6_VASC\\5k\\validation"
test_dir = "E:\MELANOMA\FENG497-MELANOMA\APPROACH2\MODEL6_VASC\\5k\\test"

# Create data loaders
image_data_loader = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, labels='inferred', label_mode='binary', color_mode='rgb', image_size=(224, 224), batch_size=32, shuffle=True)
val_data_loader = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, labels='inferred', label_mode='binary', color_mode='rgb', image_size=(224, 224), batch_size=32, shuffle=True)
test_data_loader = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, labels='inferred', label_mode='binary', color_mode='rgb', image_size=(224, 224), batch_size=32, shuffle=True)

# Load the pre-trained VGG16 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Define a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)
#optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
#optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr_schedule)
optimizer = Adam(learning_rate=lr_schedule)
# Define early stopping
# early_stopping = EarlyStopping(
 #    monitor='val_loss',
  #   patience=10,
  #   restore_best_weights=True)

# Compile the model with Binary Cross-Entropy loss
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
history = model.fit(
    image_data_loader,
    steps_per_epoch=100,  # Adjust as per your dataset
    epochs=30,
    validation_data=val_data_loader,
    validation_steps=50)  # Adjust as per your dataset
    # callbacks=[early_stopping])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data_loader)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict classes
predictions = model.predict(test_data_loader)
predictions = np.round(predictions).astype(int)

# True labels
true_classes = []
for _, labels in test_data_loader:
    true_classes.extend(labels.numpy())

# Confusion Matrix
cm = confusion_matrix(true_classes, predictions)
print("Confusion Matrix:")
print(cm)

# Classification report for precision, recall, f1-score
cr = classification_report(true_classes, predictions, target_names=['class_0', 'class_1'])  # Replace class names accordingly
print("Classification Report:")
print(cr)

# Plot confusion matrix
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)  # Assuming binary classification
plt.xticks(tick_marks, ['class_0', 'class_1'])
plt.yticks(tick_marks, ['class_0', 'class_1'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Save the model
model.save('vgg19_skin_cancer_model.keras')

import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('vgg19_skin_cancer_model.keras')

# Convert the TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to disk
tflite_model_path = 'E:\\MELANOMA\\FENG497-MELANOMA\\APPROACH2\\MODEL6_VASC\\vgg19_skin_cancer_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model saved as TensorFlow Lite model at: {tflite_model_path}")








