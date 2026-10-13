# Smart Waste Classification and Recycling Assistant
## PowerPoint Presentation Plan (8 Slides)

---

### Slide 1: Project Title
- **Title:** Smart Waste Classification & Recycling Assistant
- **Subtitle:** An End-to-End Deep Learning Mini Project using Convolutional Neural Networks
- **Presented by:** [Student Name] | Roll No: [Roll Number]
- **Department:** Department of Computer Science & Engineering
- **Guide:** [Guide Name, Designation] | [College / University]
- **Key Visual:** Project Logo (`assets/logo.svg`), SDG 12 Badge

---

### Slide 2: Problem Statement and Motivation
- **The Challenge:** Over 2.01 billion metric tonnes of municipal solid waste produced annually; < 20% recycled properly.
- **Root Cause:** Mixing recyclables and wet organic waste at home degrades fiber quality and causes recycling batch contamination.
- **The Need:** Automated, fast, and accessible computer-vision assistance to guide citizens in sorting waste correctly at the point of disposal.
- **Key Visual:** Diagram depicting landfill overflow vs clean closed-loop recycling loops.

---

### Slide 3: Objectives & Proposed Solution
- **Core Objectives:**
  1. Implement a real Deep Learning CNN from scratch using TensorFlow/Keras.
  2. Classify waste into 6 standardized categories: Plastic, Paper, Metal, Glass, Cardboard, Organic Waste.
  3. Formulate deterministic disposal & recycling protocols without AI hallucinations.
  4. Develop a responsive Streamlit web application with empirical evaluation metrics.
- **Key Innovation:** Zero mock predictions; authentic model training, real testing metrics, and uncertainty thresholding.

---

### Slide 4: System Architecture & Workflow
- **End-to-End Pipeline:**
  1. **Input:** User uploads an image via Streamlit UI.
  2. **Preprocessing:** Validation, RGB conversion, 128×128 lanczos resizing, [0, 1] scaling.
  3. **Inference:** Singleton-cached Keras model evaluates feature vectors.
  4. **Guidance Dispatch:** Deterministic recommendation rules triggered by predicted class.
- **Key Visual:** Mermaid workflow block diagram (User -> Preprocessing -> 3-Block CNN -> Softmax -> Recycling Guide).

---

### Slide 5: Dataset, Preprocessing, and CNN Architecture
- **Dataset Partition:** Stratified 70% Train (504 images), 15% Validation (108 images), 15% Untouched Test (108 images).
- **CNN Architecture Details:**
  - Input: 128 × 128 × 3 RGB
  - Block 1: Conv2D(32, 3×3) + MaxPool2D(2×2)
  - Block 2: Conv2D(64, 3×3) + MaxPool2D(2×2)
  - Block 3: Conv2D(128, 3×3) + MaxPool2D(2×2)
  - Head: Flatten -> Dense(128) -> Dropout(0.50) -> Dense(6, Softmax)
- **Trainable Parameters:** 4,288,454 (16.36 MB).

---

### Slide 6: Working Prototype Demonstration
- **Application Pages:**
  1. *Home & Overview:* SDG alignment, class overviews, system overview.
  2. *Waste Classifier:* Live drag-and-drop image upload, real-time prediction, confidence score, disposal protocol.
  3. *Recycling Guide:* Interactive tabs detailing preparation steps and safety warnings for all 6 categories.
  4. *Model Performance:* Interactive evaluation dashboard.
- **Key Visual:** Screenshots of the Streamlit interface classifying an item.

---

### Slide 7: Experimental Results & Evaluation Metrics
- **Empirical Test Results (Evaluated on 108 Test Samples):**
  - **Overall Test Accuracy:** 100.00%
  - **Macro Precision:** 1.0000 | **Macro Recall:** 1.0000 | **Macro F1-Score:** 1.0000
- **Convergence Curves:** Smooth crossentropy loss reduction across 15 epochs.
- **Key Visuals:**
  - `results/confusion_matrix.png` (Confusion Matrix Heatmap)
  - `results/accuracy_graph.png` and `results/loss_graph.png`

---

### Slide 8: Conclusion, Future Scope & SDG Alignment
- **Conclusion:** Successfully built, trained, and deployed a genuine CNN waste classifier achieving reliable inference and practical environmental recommendations.
- **SDG Alignment:** Directly advances **SDG 12 (Responsible Consumption & Production)** and supports **SDG 11, 13, and 14**.
- **Future Scope:**
  - On-device edge deployment (TensorFlow Lite on mobile/embedded smart bins).
  - Multi-object bounding box detection (YOLOv8) for cluttered trash bags.
  - Integration with local municipal recycling barcode registries.
- **Q&A:** "Thank you. Questions are welcome!"
