# 🥔 Potato Leaf Disease Detection

A deep learning project that classifies potato leaf images into **Early Blight**, **Late Blight** or **Healthy** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shantanuparekh/potato-leaf-disease-detection/blob/main/potato_disease_colab.ipynb)

## Run it in Google Colab (no installation)

1. Click the **Open in Colab** badge above.
2. **Just want to try it?** Run *Setup*, then *Option A* (loads the pretrained model), then the last cell and upload a leaf photo.
3. **Want to train it yourself?** Upload the dataset to Google Drive, follow *Option B*, then run the last cell.

> Before first use, replace `shantanuparekh` in this README and in the notebook's `GITHUB_USER` variable with your GitHub username.

## Classes

| Class | Meaning |
|---|---|
| `Potato___Early_blight` | Early Blight (*Alternaria solani*) |
| `Potato___Late_blight` | Late Blight (*Phytophthora infestans*) |
| `Potato___healthy` | Healthy leaf |

## Dataset

The dataset is **not stored in this repo** (too large). Use a potato-leaf dataset arranged as one folder per class with the three names above (e.g. the potato classes of the PlantVillage dataset). Your folder should look like:

```
dataset/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
```

Split used: **80% training / 20% testing** (plus 10% of the training part for validation during training).

## Model

4 × (Conv2D + MaxPooling) → Flatten → Dense(128) → Dropout(0.5) → Dense(3, softmax), with data augmentation (flip, rotate, zoom). About 831K trainable parameters, input size 128×128.

## Run locally

```bash
pip install -r requirements.txt

# train
python train_model.py --data_dir "path/to/dataset" --output_dir models --epochs 20

# desktop GUI (uses ./models by default; set MODEL_DIR to use another folder)
python gui_app.py
```

## Project structure

```
├── potato_disease_colab.ipynb   # Colab notebook (train + predict)
├── train_model.py               # training script
├── gui_app.py                   # Tkinter desktop GUI
├── requirements.txt
├── models/                      # trained model + class_names.json
└── docs/Potato_Model_Code_Explained.pdf   # plain-language code explanation
```

## Results

> Fill in after training: test accuracy, confusion matrix and training curves.

| Metric | Value |
|---|---|
| Test accuracy | _xx.xx %_ |

## Limitations

- The dataset may contain augmented copies of the same photos, so a random split can inflate test accuracy.
- Images are lab-style; accuracy on real field photos may be lower.
- Only 3 classes: a non-potato image will still be assigned one of them (check the confidence score).
- This is an educational project, not a substitute for expert agronomic advice.

## License

MIT - see [LICENSE](LICENSE).
