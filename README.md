# codealpha_tasks_2
Draw → Predict → Recognize! An interactive handwriting recognition tool powered by deep learning (EMNIST + MNIST).
# ✍️ Handwriting Recognition (Digits + Letters)

This project is a **handwriting recognition app** built with **TensorFlow/Keras + Tkinter**.  
It can recognize both **digits (0–9)** and **letters (A–Z)** using a **CNN trained on EMNIST + MNIST** datasets.  

---

## 🚀 Features
- Recognizes **digits (0–9)** and **uppercase letters (A–Z)**.
- First run → trains CNN model and saves as `handwriting_model.h5`.
- Later runs → loads model instantly (no retraining).
- Interactive **Tkinter GUI**: draw, predict, and clear canvas.

---

## 📦 Setup

### 1. Clone Repo
```bash
git clone https://github.com/your-username/handwriting-recognition.git
cd handwriting-recognition
2. Install Dependencies
pip install -r requirements.txt

3. Download EMNIST Dataset

Download EMNIST Letters (in .gz format) from:
👉 EMNIST Dataset (by NIST)

Extract files into the gzip/ folder:

emnist-letters-train-images-idx3-ubyte.gz
emnist-letters-train-labels-idx1-ubyte.gz
emnist-letters-test-images-idx3-ubyte.gz
emnist-letters-test-labels-idx1-ubyte.gz


(MNIST will be downloaded automatically by TensorFlow, no manual steps needed.)

▶️ Run the App
python handwriting_app.py


Draw digits/letters on the canvas.

Click Predict to see the result.

Click Clear to reset the canvas.

🧠 Model

CNN with 2 convolutional layers + dense layers

Trained on:

EMNIST Letters (A–Z)

MNIST Digits (0–9)

Classes:

0–25 → A–Z

26–35 → 0–9

📋 Requirements

requirements.txt

tensorflow
numpy
pillow

📸 Demo

📜 License

MIT License – free to use and modify.


---

⚡ Next steps for you:
1. Copy the `handwriting_app.py` I gave earlier.  
2. Add a `requirements.txt` with:


tensorflow
numpy
pillow
