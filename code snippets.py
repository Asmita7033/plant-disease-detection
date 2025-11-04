import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

interpreter = tflite.Interpreter(model_path='models/model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open('leaf.jpg').convert('RGB').resize((224,224))
arr = np.array(img).astype('float32')
arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr) # if using MobileNet preprocessing
arr = np.expand_dims(arr, 0)
interpreter.set_tensor(input_details[0]['index'], arr)
interpreter.invoke()
preds = interpreter.get_tensor(output_details[0]['index'])[0]
