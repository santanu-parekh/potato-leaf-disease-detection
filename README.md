# 🥔 Potato Leaf Disease Detection

A deep learning project that classifies potato leaf images into **Early Blight**, **Late Blight** or **Healthy** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It includes a training script and a simple desktop GUI for testing leaf photos.






https://github.com/user-attachments/assets/f21885ca-b2dd-4c07-8aab-f6c0ce2a2774






## Classes

| Class | Meaning |
|---|---|
| `Potato___Early_blight` | Early Blight (*Alternaria solani*) |
| `Potato___Late_blight` | Late Blight (*Phytophthora infestans*) |
| `Potato___healthy` | Healthy leaf |

## Installation (local)

**Requirements:** Python 3.10 or newer (check that your Python version is supported by TensorFlow) and `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/shantanuparekh/potato-leaf-disease-detection.git
cd potato-leaf-disease-detection

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset

The dataset is **not stored in this repo** (too large). Download a potato-leaf dataset and arrange it as one folder per class using the three names above (e.g. the potato classes of the PlantVillage dataset):

```
dataset/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
```

Split used: **80% training / 20% testing** (plus 10% of the training part for validation during training).

## Usage

### 1. Train the model

```bash
python train_model.py --data_dir "path/to/dataset" --output_dir models --epochs 20
```

This saves the following into `models/`:
- `potato_model.keras` - the trained model
- `class_names.json` - class order used by the model
- `training_curves.png` and `confusion_matrix.png` - result graphs

It also prints test accuracy and a classification report (precision, recall, F1).

### 2. Test with the desktop GUI

```bash
python gui_app.py
```

Click **Choose Image**, select a leaf photo, then click **Predict** to see the disease, confidence and short treatment advice.
The GUI reads the model from `./models` by default. To use a model stored elsewhere, set the `MODEL_DIR` environment variable first:

```bash
set MODEL_DIR=C:\path\to\model_folder        # Windows (cmd)
# export MODEL_DIR=/path/to/model_folder     # macOS / Linux
```

## Model

4 × (Conv2D + MaxPooling) → Flatten → Dense(128) → Dropout(0.5) → Dense(3, softmax), with data augmentation (flip, rotate, zoom). About 831K trainable parameters, input size 128×128.

## Project structure

```
├── train_model.py     # training + evaluation script
├── gui_app.py         # Tkinter desktop GUI
├── requirements.txt
├── models/            # trained model + class_names.json
└── docs/Potato_Model_Code_Explained.pdf   # plain-language code explanation
```

## Results


<img width="1374" height="1145" alt="Dark Theme Potato Confusion Matrix" src="https://github.com/user-attachments/assets/0725c4b5-ea42-4bc6-bddc-d2ac5c22f95f" />
<img width="2172" height="724" alt="Dark Theme Accuracy and Loss Charts" src="https://github.com/user-attachments/assets/2f2d920d-ac41-43cb-927b-1cab181d37a6" />



## Limitations

- The dataset may contain augmented copies of the same photos, so a random split can inflate test accuracy.
- Images are lab-style; accuracy on real field photos may be lower.
- Only 3 classes: a non-potato image will still be assigned one of them (check the confidence score).
- This is an educational project, not a substitute for expert agronomic advice.

## License

MIT - see [LICENSE](LICENSE).
