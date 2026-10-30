import sys
import numpy as np
from tensorflow import keras

IMG_SIZE = (224, 224)
CLASSES = ["cat", "dog"]  # 0 = cat, 1 = dog

model = keras.models.load_model("model/cat_dog_model.keras")

path = sys.argv[1]
img = keras.utils.load_img(path, target_size=IMG_SIZE)
arr = np.expand_dims(keras.utils.img_to_array(img), axis=0)

p = float(model.predict(arr, verbose=0)[0][0])
label = CLASSES[int(p > 0.5)]
conf = p if p > 0.5 else 1 - p
print(f"Prediction: {label} ({conf:.1%} confidence)")