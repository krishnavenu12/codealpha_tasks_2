import os
import gzip
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------
# Load EMNIST from .gz
# -----------------------
def load_idxfile(path, is_images=True):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        if is_images:
            magic, num, rows, cols = np.frombuffer(data[:16], dtype=">i4")
            return np.frombuffer(data[16:], dtype=np.uint8).reshape(num, rows, cols)
        else:
            magic, num = np.frombuffer(data[:8], dtype=">i4")
            return np.frombuffer(data[8:], dtype=np.uint8)

# Path where you saved EMNIST .gz files
emnist_path = os.path.join("gzip")

# -----------------------
# Dataset preparation
# -----------------------
def prepare_data():
    # Load EMNIST
    x_train_e = load_idxfile(os.path.join(emnist_path, "emnist-letters-train-images-idx3-ubyte.gz"))
    y_train_e = load_idxfile(os.path.join(emnist_path, "emnist-letters-train-labels-idx1-ubyte.gz"), is_images=False)
    x_test_e = load_idxfile(os.path.join(emnist_path, "emnist-letters-test-images-idx3-ubyte.gz"))
    y_test_e = load_idxfile(os.path.join(emnist_path, "emnist-letters-test-labels-idx1-ubyte.gz"), is_images=False)

    # Fix EMNIST orientation
    x_train_e = np.transpose(x_train_e, (0, 2, 1))
    x_test_e = np.transpose(x_test_e, (0, 2, 1))

    # Labels 1–26 → shift to 0–25
    y_train_e = y_train_e.copy() - 1
    y_test_e = y_test_e.copy() - 1

    # Load MNIST directly
    (x_train_m, y_train_m), (x_test_m, y_test_m) = keras.datasets.mnist.load_data()

    # Preprocess
    x_train_e = x_train_e.astype("float32") / 255.0
    x_test_e = x_test_e.astype("float32") / 255.0
    x_train_m = x_train_m.astype("float32") / 255.0
    x_test_m = x_test_m.astype("float32") / 255.0

    # Add channel dimension
    x_train_e = np.expand_dims(x_train_e, -1)
    x_test_e = np.expand_dims(x_test_e, -1)
    x_train_m = np.expand_dims(x_train_m, -1)
    x_test_m = np.expand_dims(x_test_m, -1)

    # Shift MNIST labels to 26–35
    y_train_m = y_train_m + 26
    y_test_m = y_test_m + 26

    # Combine datasets
    x_train = np.concatenate([x_train_e, x_train_m], axis=0)
    y_train = np.concatenate([y_train_e, y_train_m], axis=0)
    x_test = np.concatenate([x_test_e, x_test_m], axis=0)
    y_test = np.concatenate([y_test_e, y_test_m], axis=0)

    num_classes = 36
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), num_classes

# -----------------------
# Build CNN Model
# -----------------------
def build_model(num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -----------------------
# Load or Train Model
# -----------------------
model_path = "handwriting_model.h5"

if os.path.exists(model_path):
    print("✅ Loading saved model...")
    model = keras.models.load_model(model_path)
else:
    print("⚡ Training new model...")
    (x_train, y_train), (x_test, y_test), num_classes = prepare_data()
    model = build_model(num_classes)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128)
    model.save(model_path)
    print("💾 Model saved!")

# -----------------------
# Tkinter GUI
# -----------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwriting Recognition (Digits + Letters)")
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.clear)
        self.button_clear.pack()
        self.label_result = tk.Label(self, text="Draw a digit (0–9) or letter (A–Z)", font=("Arial", 16))
        self.label_result.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280,280), 255)
        self.draw_instance = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw_instance.ellipse([x-r,y-r,x+r,y+r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw_instance.rectangle([0,0,280,280], fill=255)
        self.label_result.config(text="Draw a digit (0–9) or letter (A–Z)")

    def predict(self):
        img = self.image.resize((28,28))
        img = ImageOps.invert(img)  # black text on white bg
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0,-1))
        pred = model.predict(img)
        class_idx = np.argmax(pred)
        if class_idx < 26:
            char = chr(class_idx + ord("A"))
        else:
            char = str(class_idx - 26)
        self.label_result.config(text=f"Prediction: {char}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
