import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

# Check if GPU is available
device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'

# Define paths to the datasets
data_dir = 'E:\\MELANOMA\\FENG497-MELANOMA\\APPROACH2\\MODEL7_DF\\model7DF_dataset'
train_dir = f'{data_dir}/training'
test_dir = f'{data_dir}/test'
val_dir = f'{data_dir}/validation'

# Define image data generator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Setup data flows
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Load the Vision Transformer model from TensorFlow Hub
hub_url = "https://tfhub.dev/google/vit_base_patch16_224/1"
base_model = hub.KerasLayer(hub_url, trainable=False, input_shape=(224, 224, 3))

# Add new classifier layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Change to one output unit with sigmoid activation

# Construct the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('model_vit', save_best_only=True, monitor='val_loss', mode='min', save_format='tf')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20,
    callbacks=[checkpoint, early_stopping, reduce_lr])

# Directly save the model in SavedModel format after training
model.save('model_vit_final', save_format='tf')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# To continue with further epochs or another training phase, one might unfreeze some layers or change learning rates, etc.
