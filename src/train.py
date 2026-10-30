import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
BATCH = 32
BASE = os.path.join("dataset", "data")
TRAIN_DIR = os.path.join(BASE, "training_set", "training_set")
TEST_DIR = os.path.join(BASE, "test_set", "test_set")

# Load data (cats = 0, dogs = 1, alphabetical order)
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="binary", validation_split=0.2, subset="training", seed=42)
val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="binary", validation_split=0.2, subset="validation", seed=42)
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="binary", shuffle=False)

print("Classes:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Model: MobileNetV2 (transfer learning)
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
# ---- Training ----
os.makedirs("model", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "model/cat_dog_model.keras", monitor="val_accuracy",
        save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True),
]

history = model.fit(train_ds, validation_data=val_ds,
                    epochs=8, callbacks=callbacks)

# ---- Final evaluation on the test set ----
loss, acc = model.evaluate(test_ds)
print(f"\nTEST accuracy: {acc:.4f}  |  TEST loss: {loss:.4f}")

# Save weights separately too (needed for the GitHub repo)
model.save_weights("model/cat_dog.weights.h5")

# Save training curves for the report
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout()
plt.savefig("model/training_curves.png", dpi=150)
print("Saved model and training curves in the model/ folder")