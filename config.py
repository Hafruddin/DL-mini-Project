"""
Configuration module for the Smart Waste Classification and Recycling Assistant.
Centralizes paths, hyperparameters, target classes, and UI configurations.
"""

from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "waste_cnn.keras"
CLASS_MAPPING_PATH = MODELS_DIR / "class_indices.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

RESULTS_DIR = BASE_DIR / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ACCURACY_GRAPH_PATH = RESULTS_DIR / "accuracy_graph.png"
LOSS_GRAPH_PATH = RESULTS_DIR / "loss_graph.png"

ASSETS_DIR = BASE_DIR / "assets"
CATEGORY_ICONS_DIR = ASSETS_DIR / "category_icons"

# Image Preprocessing Specifications
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Target Classes (Exactly 6 required classes in standard alphabetical order)
TARGET_CLASSES = [
    "Cardboard",
    "Glass",
    "Metal",
    "Organic Waste",
    "Paper",
    "Plastic"
]

# Kaggle dataset class name mapping to 6 target classes
# Typical Garbage Classification dataset classes: cardboard, glass, metal, paper, plastic, trash
# If trash/organic datasets are used, maps consistently:
SOURCE_TO_TARGET_MAP = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Organic Waste",  # In standard garbage classification benchmarks, 'trash' contains organic food remnants & residual domestic waste
    "organic": "Organic Waste",
    "organic waste": "Organic Waste",
    "biological": "Organic Waste"
}

# Training Hyperparameters
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Prediction & Inference Confidence Threshold
CONFIDENCE_THRESHOLD = 0.60  # Below this, model raises an uncertainty/caution advisory

# UI Theme Colors (Restrained environmental design system)
COLOR_PRIMARY_FOREST_GREEN = "#245B45"
COLOR_SECONDARY_GREEN = "#477A5D"
COLOR_LIGHT_SAGE = "#EAF1E8"
COLOR_BACKGROUND = "#F7F8F5"
COLOR_MAIN_TEXT = "#202820"
COLOR_SECONDARY_TEXT = "#687168"
COLOR_BORDER = "#E0E5DD"
COLOR_WHITE = "#FFFFFF"
