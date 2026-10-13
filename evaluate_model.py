"""
Model evaluation script for the Smart Waste Classification CNN.
Evaluates the trained model strictly on the untouched test dataset.
Generates confusion matrix, per-class classification report, and metrics.json.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import tensorflow as tf
from tensorflow import keras

from config import (
    TEST_DIR,
    MODEL_PATH,
    CLASS_MAPPING_PATH,
    TARGET_CLASSES,
    RESULTS_DIR,
    METRICS_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    IMG_HEIGHT,
    IMG_WIDTH
)
from preprocessing import load_and_preprocess_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate():
    logger.info("--- MODEL EVALUATION ON UNTOUCHED TEST DATASET ---")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Train the model first.")

    # Load Model
    logger.info(f"Loading trained CNN model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    # Load class mapping
    if CLASS_MAPPING_PATH.exists():
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
            # Ensure keys are integers
            class_mapping = {int(k): v for k, v in class_mapping.items()}
    else:
        class_mapping = {i: c for i, c in enumerate(TARGET_CLASSES)}

    # Collect true labels and predictions
    y_true = []
    y_pred = []
    y_scores = []
    test_image_paths = []

    logger.info("Evaluating test set images...")
    for class_idx, class_name in enumerate(TARGET_CLASSES):
        class_folder = TEST_DIR / class_name
        if not class_folder.exists():
            continue
        
        image_files = sorted([f for f in class_folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        for img_file in image_files:
            try:
                processed = load_and_preprocess_image(img_file)
                batch_tensor = np.expand_dims(processed, axis=0)
                pred_probs = model.predict(batch_tensor, verbose=0)[0]
                pred_label = int(np.argmax(pred_probs))

                y_true.append(class_idx)
                y_pred.append(pred_label)
                y_scores.append(pred_probs)
                test_image_paths.append(str(img_file))
            except Exception as e:
                logger.warning(f"Error processing test image {img_file.name}: {e}")

    if len(y_true) == 0:
        raise RuntimeError("No valid test images found in test directory.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Compute Global Metrics
    test_accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Compute Per-Class Metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(TARGET_CLASSES))), zero_division=0
    )

    class_metrics = {}
    for i, cname in enumerate(TARGET_CLASSES):
        class_metrics[cname] = {
            "precision": round(float(per_class_precision[i]), 4),
            "recall": round(float(per_class_recall[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_support[i])
        }

    # Generate Scikit-Learn Classification Report
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_CLASSES,
        digits=4,
        zero_division=0
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save Text Report
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("SMART WASTE CLASSIFICATION — TEST EVALUATION REPORT\n")
        f.write(f"Evaluation Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Test Samples: {len(y_true)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
    logger.info(f"Classification report saved to {CLASSIFICATION_REPORT_PATH}")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_CLASSES))))

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES,
        cbar=True,
        linewidths=0.5,
        linecolor="#E0E5DD",
        square=True
    )
    plt.title("Confusion Matrix — Waste Classification CNN", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Category", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth Category", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix heatmap saved to {CONFUSION_MATRIX_PATH}")

    # Save Structured Metrics JSON
    metrics_summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "test_dataset_size": len(y_true),
        "overall_test_accuracy": round(test_accuracy, 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1_score": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_precision), 4),
        "weighted_recall": round(float(weighted_recall), 4),
        "weighted_f1_score": round(float(weighted_f1), 4),
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
        "target_classes": TARGET_CLASSES
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info(f"Metrics JSON saved to {METRICS_PATH}")

    print("\n" + "=" * 55)
    print("TEST EVALUATION COMPLETED")
    print(f"Overall Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"Macro Precision:   {macro_precision * 100:.2f}%")
    print(f"Macro Recall:      {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print("=" * 55 + "\n")

    return metrics_summary


if __name__ == "__main__":
    evaluate()
