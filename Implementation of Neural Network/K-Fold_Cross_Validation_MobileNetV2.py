import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Define data directories and constants
train_data_dir = 'C:\\Users\\nilay\\Desktop\\Dataset\\MainDataset_224\\Training'
image_size = (224, 224)
input_shape = image_size + (3,)
batch_size = 16
num_classes = 25
num_folds = 5

# Load all data and labels
datagen = ImageDataGenerator(rescale=1./255)
data_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Concatenate data for K-Fold
all_data, all_labels = next(data_generator)
for _ in range(1, data_generator.__len__()):
    imgs, labels = next(data_generator)
    all_data = np.vstack((all_data, imgs))
    all_labels = np.vstack((all_labels, labels))

# K-Fold Cross Validation model evaluation
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train, test in kfold.split(all_data, all_labels):
    # Define the model architecture
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=100000, decay_rate=0.96, staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit data to model
    history = model.fit(all_data[train], all_labels[train],
                        epochs=20,
                        validation_data=(all_data[test], all_labels[test]))

    # Generate generalization metrics
    scores = model.evaluate(all_data[test], all_labels[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    fold_no += 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
