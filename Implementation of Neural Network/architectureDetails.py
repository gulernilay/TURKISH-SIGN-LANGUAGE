import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="C:\Users\nilay\Downloads\detect (2).tflite")
interpreter.allocate_tensors()

# Get the model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
all_layers_details = interpreter.get_tensor_details() 

for layer in all_layers_details:
    print(layer['index'], layer['name'], layer['shape'])  # This will print out each layer's details
