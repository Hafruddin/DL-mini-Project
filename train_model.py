"""
Standalone training script for the Smart Waste Classification CNN model.
Handles dataset loading, training augmentation, model training with callbacks,
saving of weights & artifacts, and visualization plots.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TARGET_CLASSES,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    INITIAL_EPOCHS,
    RANDOM_SEED,
    MODEL_PATH,
    CLASS_MAPPING_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    ACCURACY_GRAPH_PATH,
    LOSS_GRAPH_PATH,
    COLOR_PRIMARY_FOREST_GREEN,
    COLOR_SECONDARY_GREEN
)
from cnn_model import build_waste_cnn, get_data_augmentation_layer
from dataset_loader import verify_dataset_integrity, count_split_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train():
    logger.info("--- SMART WASTE CLASSIFICATION MODEL TRAINING ---")

    # Step 1: Verify Dataset
    is_valid, msg = verify_dataset_integrity()
    if not is_valid:
        raise RuntimeError(f"Dataset verification failed: {msg}")
    logger.info(f"Dataset Status: {msg}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: Load Training and Validation Datasets with deterministic ordering
    logger.info("Loading training and validation datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=TARGET_CLASSES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        class_names=TARGET_CLASSES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False
    )

    # Step 3: Rescale images to [0, 1] and apply Training Augmentation
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    augmentation_layer = get_data_augmentation_layer()

    # Preprocessing pipelines
    train_ds = train_ds.map(lambda x, y: (augmentation_layer(normalization_layer(x), training=True), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Performance optimization (AUTOTUNE prefetch)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Step 4: Build Model
    logger.info("Building Convolutional Neural Network...")
    model = build_waste_cnn(num_classes=len(TARGET_CLASSES))
    model.summary()

    # Step 5: Configure Callbacks
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Step 6: Train Model
    logger.info(f"Initiating training for {INITIAL_EPOCHS} epochs (batch size: {BATCH_SIZE})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback, reduce_lr_callback],
        verbose=1
    )

    # Step 7: Guarantee Best Model is Saved
    model.save(MODEL_PATH)
    logger.info(f"Model saved successfully to: {MODEL_PATH}")

    # Step 8: Save Class Index Mapping
    class_mapping = {index: class_name for index, class_name in enumerate(TARGET_CLASSES)}
    with open(CLASS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4)
    logger.info(f"Class mapping saved to: {CLASS_MAPPING_PATH}")

    # Step 9: Save Model Metadata
    counts = count_split_images()
    metadata = {
        "model_architecture": "Custom 3-Block CNN (Conv2D -> MaxPool2D -> Dropout -> Dense)",
        "input_shape": [IMG_HEIGHT, IMG_WIDTH, 3],
        "target_classes": TARGET_CLASSES,
        "epochs_trained": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "train_samples": sum(counts["train"].values()),
        "validation_samples": sum(counts["validation"].values()),
        "test_samples": sum(counts["test"].values()),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to: {MODEL_METADATA_PATH}")

    # Step 10: Generate and Save Training Curves
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy Graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["accuracy"], marker="o", color=COLOR_PRIMARY_FOREST_GREEN, label="Training Accuracy", linewidth=2)
    plt.plot(epochs_range, history.history["val_accuracy"], marker="s", color="#D97706", label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy Across Epochs", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(ACCURACY_GRAPH_PATH, dpi=300)
    plt.close()
    logger.info(f"Saved accuracy graph: {ACCURACY_GRAPH_PATH}")

    # Loss Graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["loss"], marker="o", color=COLOR_PRIMARY_FOREST_GREEN, label="Training Loss", linewidth=2)
    plt.plot(epochs_range, history.history["val_loss"], marker="s", color="#DC2626", label="Validation Loss", linewidth=2)
    plt.title("Model Loss Convergence Across Epochs", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Sparse Categorical Crossentropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(LOSS_GRAPH_PATH, dpi=300)
    plt.close()
    logger.info(f"Saved loss graph: {LOSS_GRAPH_PATH}")

    logger.info("Training pipeline completed successfully.")
    return metadata


if __name__ == "__main__":
    train()
