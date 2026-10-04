"""
Potato Leaf Disease Detector - Simple GUI (Tkinter)

Install requirements first (tkinter comes with Python on Windows):
    pip install pillow numpy tensorflow
"""

import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

# ----------------------------------------------------------------------
# CONFIG  (must match the training script)
# ----------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
MODEL_PATH = os.path.join(OUTPUT_DIR, "potato_model.keras")
CLASSES_PATH = os.path.join(OUTPUT_DIR, "class_names.json")
IMG_SIZE = (128, 128)
LOW_CONFIDENCE = 70.0   # below this %, warn the user

# Friendly names and advice for each class
DISEASE_INFO = {
    "Potato___Early_blight": {
        "name": "Early Blight",
        "color": "#d9822b",
        "advice": "Caused by the fungus Alternaria solani.\n"
                  "- Remove and destroy infected leaves\n"
                  "- Apply a recommended fungicide (e.g. mancozeb / chlorothalonil)\n"
                  "- Rotate crops and avoid overhead watering",
    },
    "Potato___Late_blight": {
        "name": "Late Blight",
        "color": "#c0392b",
        "advice": "Caused by Phytophthora infestans - spreads very fast.\n"
                  "- Remove infected plants immediately\n"
                  "- Apply a suitable fungicide at once\n"
                  "- Improve air flow and avoid wet foliage",
    },
    "Potato___healthy": {
        "name": "Healthy Leaf",
        "color": "#27ae60",
        "advice": "No disease detected.\n"
                  "- Keep monitoring the crop regularly\n"
                  "- Maintain proper watering and fertilization",
    },
}

# ----------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
    raise SystemExit(f"Model files not found in {OUTPUT_DIR}. Run train_model.py first.")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH) as f:
    class_names = json.load(f)
print("Model loaded. Classes:", class_names)


def predict(image_path):
    """Return (class_name, confidence_percent, all_probabilities)."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")          # 0-255; model rescales internally
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]) * 100, probs


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Potato Leaf Disease Detector")
        self.geometry("560x720")
        self.configure(bg="#f4f6f8")
        self.resizable(False, False)

        self.photo = None
        self.image_path = None

        tk.Label(self, text="Potato Leaf Disease Detector",
                 font=("Segoe UI", 18, "bold"), bg="#f4f6f8", fg="#2c3e50").pack(pady=(18, 4))
        tk.Label(self, text="Upload a potato leaf image to check its health",
                 font=("Segoe UI", 10), bg="#f4f6f8", fg="#7f8c8d").pack(pady=(0, 12))

        # Image preview box
        self.preview = tk.Label(self, text="No image selected", bg="#dfe6e9",
                                fg="#636e72", width=40, height=14, font=("Segoe UI", 10))
        self.preview.pack(pady=6)

        # Buttons
        btn_frame = tk.Frame(self, bg="#f4f6f8")
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="Choose Image", command=self.choose_image,
                  font=("Segoe UI", 11, "bold"), bg="#3498db", fg="white",
                  padx=16, pady=6, relief="flat", cursor="hand2").grid(row=0, column=0, padx=8)
        self.predict_btn = tk.Button(btn_frame, text="Predict", command=self.run_prediction,
                                     font=("Segoe UI", 11, "bold"), bg="#27ae60", fg="white",
                                     padx=22, pady=6, relief="flat", cursor="hand2",
                                     state="disabled")
        self.predict_btn.grid(row=0, column=1, padx=8)

        # Result area
        self.result_label = tk.Label(self, text="", font=("Segoe UI", 16, "bold"),
                                     bg="#f4f6f8")
        self.result_label.pack(pady=(10, 0))
        self.conf_label = tk.Label(self, text="", font=("Segoe UI", 11),
                                   bg="#f4f6f8", fg="#2c3e50")
        self.conf_label.pack()
        self.warn_label = tk.Label(self, text="", font=("Segoe UI", 9, "italic"),
                                   bg="#f4f6f8", fg="#c0392b")
        self.warn_label.pack()

        self.prob_label = tk.Label(self, text="", font=("Consolas", 9), bg="#f4f6f8",
                                   fg="#636e72", justify="left")
        self.prob_label.pack(pady=4)

        self.advice_label = tk.Label(self, text="", font=("Segoe UI", 10), bg="#ffffff",
                                     fg="#2c3e50", justify="left", wraplength=480,
                                     padx=12, pady=10, anchor="w")
        self.advice_label.pack(pady=8, padx=30, fill="x")

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select a leaf image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((300, 260))
            self.photo = ImageTk.PhotoImage(img)
            self.preview.config(image=self.photo, text="", width=300, height=260)
            self.image_path = path
            self.predict_btn.config(state="normal")
            self.clear_results()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")

    def clear_results(self):
        self.result_label.config(text="")
        self.conf_label.config(text="")
        self.warn_label.config(text="")
        self.prob_label.config(text="")
        self.advice_label.config(text="")

    def run_prediction(self):
        if not self.image_path:
            return
        try:
            cls, conf, probs = predict(self.image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")
            return

        info = DISEASE_INFO.get(cls, {"name": cls, "color": "#2c3e50", "advice": ""})
        self.result_label.config(text=info["name"], fg=info["color"])
        self.conf_label.config(text=f"Confidence: {conf:.2f}%")
        self.warn_label.config(
            text="Low confidence - image may be unclear or not a potato leaf."
            if conf < LOW_CONFIDENCE else "")

        lines = [f"{DISEASE_INFO.get(c, {'name': c})['name']:<14} {p * 100:6.2f}%"
                 for c, p in zip(class_names, probs)]
        self.prob_label.config(text="\n".join(lines))
        self.advice_label.config(text=info["advice"])


if __name__ == "__main__":
    App().mainloop()
