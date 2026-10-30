import tensorflow as tf
import tf2onnx
from tensorflow import keras

model = keras.models.load_model("model/cat_dog_model.keras")

@tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input")])
def serve(x):
    return model(x, training=False)

tf2onnx.convert.from_function(
    serve,
    input_signature=serve.input_signature,
    opset=13,
    output_path="model/cat_dog_model.onnx",
)
print("Saved model/cat_dog_model.onnx")