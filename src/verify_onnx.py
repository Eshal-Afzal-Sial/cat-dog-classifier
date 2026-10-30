import sys
import numpy as np
import onnxruntime as ort
from tensorflow import keras

path = sys.argv[1]
img = keras.utils.load_img(path, target_size=(224, 224))
arr = np.expand_dims(keras.utils.img_to_array(img), axis=0).astype("float32")

keras_model = keras.models.load_model("model/cat_dog_model.keras")
p_keras = float(keras_model.predict(arr, verbose=0)[0][0])

sess = ort.InferenceSession("model/cat_dog_model.onnx")
input_name = sess.get_inputs()[0].name
p_onnx = float(sess.run(None, {input_name: arr})[0][0][0])

print(f"Keras: {p_keras:.6f}")
print(f"ONNX:  {p_onnx:.6f}")
print(f"Difference: {abs(p_keras - p_onnx):.8f}")