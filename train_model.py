"""
Potato Leaf Disease Classification - Training Script
Classes: Potato___Early_blight, Potato___Late_blight, Potato___healthy
Split : 80% training / 20% testing

Usage:
    pip install -r requirements.txt
    python train_model.py --data_dir "path/to/dataset" --output_dir models --epochs 20
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train the potato leaf disease CNN")
parser.add_argument("--data_dir", required=True,
                    help="Folder containing the class folders (Potato___Early_blight, ...)")
parser.add_argument("--output_dir", default="models",
                    help="Where to save the model, class names and graphs (default: ./models)")
parser.add_argument("--epochs", type=int, default=20)
args, _ = parser.parse_known_args()

DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only these folders are used as classes (ignores any other folder inside DATA_DIR)
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = args.epochs
SEED = 42

# ----------------------------------------------------------------------
# 2. LOAD DATA  (80% train / 20% test)
# ----------------------------------------------------------------------
train_full = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",      # this is our 20% TEST set
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=CLASS_NAMES,
    # shuffle must stay True (default) with the SAME seed as above, so files are
    # shuffled before the 80/20 split and every class appears in both sets
)

class_names = train_full.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Save class names (needed later for prediction / the app)
with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)

# Take 10% of the 80% training data as validation (used during training only)
n_batches = tf.data.experimental.cardinality(train_full).numpy()
val_batches = max(1, int(0.10 * n_batches))
val_ds = train_full.take(val_batches)
train_ds = train_full.skip(val_batches)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ----------------------------------------------------------------------
# 3. MODEL (CNN with data augmentation)
# ----------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    data_augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ----------------------------------------------------------------------
# 4. TRAIN
# ----------------------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ----------------------------------------------------------------------
# 5. EVALUATE ON THE 20% TEST SET
# ----------------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ----------------------------------------------------------------------
# 6. SAVE MODEL + PLOTS
# ----------------------------------------------------------------------
model.save(os.path.join(OUTPUT_DIR, "potato_model.keras"))

# Accuracy / loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (20% test set)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

print(f"\nAll files saved in: {OUTPUT_DIR}")
