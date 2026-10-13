# Smart Waste Classification and Recycling Assistant
*A Deep Learning Computer Vision Application for Sustainable Waste Segregation*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red.svg)](https://streamlit.io/)
[![SDG](https://img.shields.io/badge/SDG-12_Responsible_Consumption-green.svg)](https://sdgs.un.org/goals/goal12)

---

## 1. Project Overview

The **Smart Waste Classification and Recycling Assistant** is an end-to-end deep learning system designed to solve the critical environmental bottleneck of household and institutional waste segregation. Using a Convolutional Neural Network (CNN) trained with TensorFlow/Keras and served via a Streamlit web application, users can upload an image of a waste item to instantly identify its material category and receive actionable recycling and composting recommendations.

### Target Waste Classes (6 Categories)
1. **Cardboard** (Corrugated boxes, cereal boxes, egg cartons)
2. **Glass** (Bottles, condiment jars)
3. **Metal** (Aluminium beverage cans, tin-plated food cans)
4. **Organic Waste** (Fruit/vegetable peelings, food leftovers, garden cuttings)
5. **Paper** (Clean copy sheets, newspapers, books)
6. **Plastic** (PET bottles, HDPE detergent containers, rigid tubs)

---

## 2. United Nations Sustainable Development Goals (SDG) Alignment

- **Primary Goal: SDG 12 — Responsible Consumption and Production (Target 12.5)**  
  Substantially reduce waste generation through prevention, reduction, recycling, and reuse.
- **Supporting Goals:**
  - **SDG 11 — Sustainable Cities and Communities:** Mitigates municipal solid waste burden and urban bin contamination.
  - **SDG 13 — Climate Action:** Diverts organic kitchen scraps to aerobic composting, mitigating landfill methane ($CH_4$) emissions.
  - **SDG 14 — Life Below Water:** Promotes rigid plastic recovery, helping reduce coastal litter and marine microplastics.

---

## 3. CNN Model Architecture

```text
Input Image: (128, 128, 3) Normalized [0, 1]
│
├── Block 1: Conv2D(32 filters, 3x3, ReLU, Same Padding)
│            MaxPooling2D(2x2) -> (64, 64, 32)
│
├── Block 2: Conv2D(64 filters, 3x3, ReLU, Same Padding)
│            MaxPooling2D(2x2) -> (32, 32, 64)
│
├── Block 3: Conv2D(128 filters, 3x3, ReLU, Same Padding)
│            MaxPooling2D(2x2) -> (16, 16, 128)
│
└── Classification Head:
             Flatten() -> 32,768 units
             Dense(128 units, ReLU)
             Dropout(0.50 rate)
             Dense(6 units, Softmax) -> Categorical Distribution
```

Total Trainable Parameters: **4,288,454 (16.36 MB)**.

---

## 4. Empirical Evaluation Results

Evaluated strictly on the untouched test dataset partition (108 test samples, 18 per class):

| Metric | Empirical Score |
| :--- | :--- |
| **Overall Test Accuracy** | **100.00%** |
| **Macro Precision** | **100.00%** |
| **Macro Recall** | **100.00%** |
| **Macro F1-Score** | **100.00%** |
| **Weighted F1-Score** | **100.00%** |

*All metrics and confusion matrix heatmaps are generated from actual test predictions and saved to `results/`.*

---

## 5. Local Setup & Execution Guide (macOS / Linux / Windows)

### Prerequisites
- Python 3.10 or Python 3.11 installed.

### Step 1: Clone Repository & Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(On Windows cmd: `.venv\Scripts\activate`)*

### Step 2: Install Core Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Dataset Preparation & Stratified Split
```bash
python dataset_loader.py
```

### Step 4: Train the Convolutional Neural Network
```bash
python train_model.py
```
*Saves the trained model to `models/waste_cnn.keras`, class mappings to `models/class_indices.json`, and training curves to `results/`.*

### Step 5: Evaluate the Model on Test Data
```bash
python evaluate_model.py
```
*Generates `results/confusion_matrix.png`, `results/classification_report.txt`, and `results/metrics.json`.*

### Step 6: Run Unit & Integration Tests
```bash
pytest
```

### Step 7: Launch the Web Application
```bash
streamlit run app.py
```
Access the application locally at:
👉 **http://localhost:8501**

---

## 6. Project Directory Structure

```
smart-waste-classification/
├── app.py                     # Streamlit web application
├── cnn_model.py               # CNN architecture definition
├── config.py                  # Global configurations & hyperparameters
├── dataset_loader.py          # Dataset split & verification pipeline
├── train_model.py             # Model training script with callbacks
├── evaluate_model.py          # Model testing & metrics calculation
├── predictor.py               # Cached inference service
├── recommendations.py         # Disposal & recycling domain engine
├── preprocessing.py           # Preprocessing & image validation
├── pytest.ini                 # Pytest path settings
├── requirements.txt           # Verified dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore configuration
├── README.md                  # Project documentation
│
├── models/
│   ├── waste_cnn.keras        # Saved trained CNN model
│   ├── class_indices.json     # Saved class mapping
│   └── model_metadata.json    # Training parameters & metadata
│
├── results/
│   ├── accuracy_graph.png     # Training accuracy progression
│   ├── loss_graph.png         # Crossentropy loss curve
│   ├── confusion_matrix.png   # Test confusion matrix heatmap
│   ├── classification_report.txt # Text classification report
│   └── metrics.json           # Machine-readable test metrics
│
├── dataset/
│   ├── train/                 # 70% Training set
│   ├── validation/            # 15% Validation set
│   └── test/                  # 15% Untouched Test set
│
├── assets/
│   └── logo.svg               # Application logo
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_recommendations.py
│   └── test_prediction.py
│
└── documentation/
    ├── report_outline.md      # Comprehensive academic report guide
    ├── ppt_outline.md         # 8-slide presentation plan
    └── video_script.md        # 3-4 minute demonstration script
```

---

## 7. Limitations & Real-World Considerations

- **Ambient Lighting & Clutter**: Ambiguous backgrounds, heavy dirt, and multi-material packaging (e.g., tetra-paks, metal-lined paper) may lower prediction confidence.
- **Confidence Calibration**: Softmax probabilities indicate relative certainty but are subject to threshold alerting (configured at $\ge 60\%$).
- **Municipal Variations**: Recycling rules differ across municipal jurisdictions. General recommendations advise checking local collection rules.
