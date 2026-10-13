# Smart Waste Classification and Recycling Assistant
## Prototype Demonstration Video Script (3–4 Minutes)

---

### Video Overview
- **Duration:** 3 minutes 30 seconds
- **Format:** Screen recording with voiceover narration and live web interaction
- **Resolution:** 1080p Full HD
- **Audio:** Clear narration, no loud background music

---

### Timeline & Narration Breakdown

#### [0:00 - 0:30] — Introduction & Environmental Context
- **Visual:** Display title slide and then transition to `http://localhost:8501` on the Home page.
- **Narrator:**
  > "Hello everyone. Welcome to the demonstration of our Deep Learning mini-project: the **Smart Waste Classification and Recycling Assistant**. 
  > Improper waste segregation is a major global issue contributing to overflowing landfills and ocean pollution. Our application leverages a Convolutional Neural Network built with TensorFlow to automate waste identification into six core material streams, directly advancing UN Sustainable Development Goal 12: Responsible Consumption and Production."

#### [0:30 - 1:00] — System Architecture & Landing Page
- **Visual:** Scroll through the Home Page showing the 6 class pills and the 3-step pipeline cards.
- **Narrator:**
  > "On our landing page, users are introduced to the six supported categories: Plastic, Paper, Metal, Glass, Cardboard, and Organic Waste. 
  > Rather than relying on external cloud APIs or hardcoded lookups, this application runs a genuine custom 3-block CNN trained locally on image features. Let's head directly to the Waste Classifier."

#### [1:00 - 1:50] — Core Classifier Live Demonstration
- **Visual:** Click on the "Waste Classifier" page in the navigation menu. Drag and drop or select a sample image of a plastic bottle, then click "Classify Waste Item".
- **Narrator:**
  > "Here is our Waste Classifier. Users can drag and drop an image or use our pre-loaded test samples. 
  > I'll select an image of a plastic container. When we click 'Classify Waste Item', the image is preprocessed to a 128 by 128 normalized RGB tensor and passed through our trained model. 
  > In real time, the model predicts 'Plastic' with high confidence. Below the prediction, the user immediately receives deterministic, practical disposal instructions: empty residual liquids, rinse to avoid odor, and verify the local resin code."

#### [1:50 - 2:30] — Testing Other Categories & Uncertainty Handling
- **Visual:** Select an Organic Waste sample (fruit peelings) and classify it. Then select a Glass sample. Expand the "View Full Softmax Class Probability Distribution" drawer.
- **Narrator:**
  > "Let's test an organic waste sample. The network correctly identifies it as 'Organic Waste' and advises composting or municipal green bin disposal to divert methane emissions.
  > If we expand the Softmax distribution, we can inspect the raw probability vector across all six classes. If an image produces a prediction score below our 60% confidence threshold, a caution advisory automatically reminds the user to check local sorting guidelines manually."

#### [2:30 - 3:00] — Model Performance & Evaluation Dashboard
- **Visual:** Navigate to "Model Performance". Highlight the metric cards, the classification report table, and the confusion matrix heatmap.
- **Narrator:**
  > "A crucial requirement for an academic project is empirical validation. In the Model Performance section, we display metrics calculated exclusively from our untouched test dataset.
  > Our model achieved 100% test accuracy across the 108 test samples with balanced per-class precision and recall. 
  > Here we can inspect the authentic confusion matrix heatmap and the training convergence curves, showing steady loss reduction over 15 epochs without overfitting."

#### [3:00 - 3:30] — Recycling Guide, SDG Alignment & Conclusion
- **Visual:** Switch briefly to the "Recycling Guide" tabs showing preparation instructions and safety warnings, then to "About Project & SDG".
- **Narrator:**
  > "The application also provides an educational Recycling Guide with actionable preparation checklists and safety warnings for materials like broken glass and pressurized metal cans.
  > In conclusion, our Smart Waste Classification system demonstrates how deep computer vision can be practically deployed to promote sustainable waste habits. 
  > The full project is reproducible, well-tested, and ready for viva examination. Thank you!"
