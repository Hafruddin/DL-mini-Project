"""
Smart Waste Classification and Recycling Assistant
College Deep Learning Mini Project Web Application built with Streamlit.
Integrates trained CNN inference, real metrics evaluation, and structured recycling guides.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path
from PIL import Image
import streamlit as st

from config import (
    MODEL_PATH,
    METRICS_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    ACCURACY_GRAPH_PATH,
    LOSS_GRAPH_PATH,
    TARGET_CLASSES,
    COLOR_PRIMARY_FOREST_GREEN,
    COLOR_SECONDARY_GREEN,
    COLOR_LIGHT_SAGE,
    COLOR_BACKGROUND,
    COLOR_MAIN_TEXT,
    COLOR_SECONDARY_TEXT,
    COLOR_BORDER,
    COLOR_WHITE,
    CONFIDENCE_THRESHOLD
)
from predictor import predict_waste, get_loaded_model
from recommendations import RECOMMENDATIONS, get_recommendation

# Page Configuration
st.set_page_config(
    page_title="Smart Waste Classifier & Recycling Assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Environmental CSS Styling (Modern Sleek Dark / Black Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean container cards in dark mode */
    .env-card {
        background: #182234;
        border: 1px solid #2A3B53;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 18px;
        color: #F1F5F9;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    
    /* Header branding */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding-bottom: 12px;
        border-bottom: 1px solid #2A3B53;
        margin-bottom: 20px;
    }
    
    .brand-title {
        color: #34D399;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }
    
    .brand-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin: 0;
    }
    
    /* Category tag pills */
    .cat-pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 600;
        background-color: #064E3B;
        color: #6EE7B7;
        border: 1px solid #059669;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Metric boxes */
    .metric-box {
        background: #182234;
        border: 1px solid #2A3B53;
        border-radius: 6px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #34D399;
    }
    .metric-label {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Primary buttons */
    .stButton > button {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
    }
    .stButton > button:hover {
        background-color: #10B981 !important;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Navigation
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span style="font-size: 28px;">♻️</span>
        <div>
            <div style="font-weight: 700; font-size: 16px; color: {COLOR_PRIMARY_FOREST_GREEN};">EcoClassify</div>
            <div style="font-size: 11px; color: {COLOR_SECONDARY_TEXT};">Waste AI & Recycling</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation Menu",
        [
            "Home & Overview",
            "Waste Classifier",
            "Recycling Guide",
            "Model Performance",
            "About Project & SDG"
        ],
        index=1  # Default to Classifier
    )

    st.markdown("---")
    st.markdown("### System Status")
    if MODEL_PATH.exists():
        st.success("CNN Model: Loaded & Ready")
    else:
        st.error("CNN Model: Not Found (Run `python train_model.py`)")

    st.caption("College Deep Learning Mini Project\nTensorFlow • Streamlit • SDG 12")


# PAGE 1: HOME & OVERVIEW
if page == "Home & Overview":
    st.markdown(f"""
    <div class="brand-header">
        <span style="font-size: 38px;">♻️</span>
        <div>
            <h1 class="brand-title">Know Your Waste. Make the Right Choice.</h1>
            <p class="brand-subtitle">Smart Waste Classification and Recycling Assistant powered by Convolutional Neural Networks.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([3, 2])
    with col_h1:
        st.markdown("""
        ### Environmental Challenge & Purpose
        Improper waste segregation at the source is one of the leading contributors to landfill saturation,
        aquatic microplastic pollution, and avoidable greenhouse gas emissions. 
        
        This application uses deep computer vision to identify waste categories from uploaded photos,
        providing instant, responsible recycling and composting instructions aligned with municipal best practices.
        """)
        
        st.markdown("#### Supported Waste Classes")
        pills_html = "".join([f"<span class='cat-pill'>{c}</span>" for c in TARGET_CLASSES])
        st.markdown(pills_html, unsafe_allow_html=True)

    with col_h2:
        st.markdown(f"""
        <div class="env-card">
            <h4 style="margin-top:0; color: {COLOR_PRIMARY_FOREST_GREEN};">Quick Actions</h4>
            <p style="font-size: 13px; color: {COLOR_SECONDARY_TEXT};">
                Ready to classify waste or explore material disposal protocols?
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Waste Classifier ➔", use_container_width=True):
            st.session_state["nav_page"] = "Waste Classifier"
            st.rerun()

    st.markdown("---")
    st.markdown("### How the Deep Learning Classifier Operates")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(f"""
        <div class="env-card">
            <div style="font-size: 20px;">📷 <b>1. Image Ingestion</b></div>
            <p style="font-size: 13px; color: {COLOR_SECONDARY_TEXT}; margin-top: 8px;">
                User uploads a photo. The image is safely converted to RGB, rescaled to 128×128 pixels, and normalized to [0, 1].
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        st.markdown(f"""
        <div class="env-card">
            <div style="font-size: 20px;">🧠 <b>2. CNN Feature Extraction</b></div>
            <p style="font-size: 13px; color: {COLOR_SECONDARY_TEXT}; margin-top: 8px;">
                Three 2D Convolutional blocks with MaxPooling extract hierarchical edges, textures, silhouettes, and material patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_c3:
        st.markdown(f"""
        <div class="env-card">
            <div style="font-size: 20px;">📋 <b>3. Guidance Dispatch</b></div>
            <p style="font-size: 13px; color: {COLOR_SECONDARY_TEXT}; margin-top: 8px;">
                A Softmax dense head yields class probabilities, triggering targeted recycling rules and disposal recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)


# PAGE 2: WASTE CLASSIFIER (CORE PAGE)
elif page == "Waste Classifier":
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <h2 class="brand-title" style="font-size: 22px;">Image-Based Waste Classifier</h2>
            <p class="brand-subtitle">Upload a photograph of packaging or discarded item to detect material category.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.warning("""
        **Model File Not Found**  
        The trained CNN model (`models/waste_cnn.keras`) is currently missing.  
        Please execute the following command in your terminal to train and save the model:  
        ```bash
        python train_model.py
        ```
        """)
        st.stop()

    upload_col, preview_col = st.columns([1, 1], gap="large")

    with upload_col:
        st.markdown(f"""
        <div style="font-size: 14px; font-weight: 600; margin-bottom: 6px;">Select or Drag & Drop Image</div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a waste image (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        st.caption("Accepted formats: JPG, JPEG, PNG, WEBP. Maximum recommended file size: 10 MB.")

        # Preset test samples for quick evaluation
        st.markdown("---")
        st.markdown("<small><b>Or test instantly with pre-loaded samples:</b></small>", unsafe_allow_html=True)
        sample_cols = st.columns(3)
        for i, cname in enumerate(TARGET_CLASSES):
            col_target = sample_cols[i % 3]
            sample_file = list((Path("dataset/test") / cname).glob("*.*"))
            if sample_file:
                if col_target.button(f"Try {cname}", key=f"btn_sample_{cname}", use_container_width=True):
                    st.session_state["selected_sample_path"] = str(sample_file[0])

    with preview_col:
        image_to_process = None
        
        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file)
                st.session_state.pop("selected_sample_path", None)
                st.image(image_to_process, caption="Uploaded Item Preview", use_container_width=True)
            except Exception as e:
                st.error(f"Error reading uploaded image: {e}")
        elif "selected_sample_path" in st.session_state and Path(st.session_state["selected_sample_path"]).exists():
            image_to_process = Image.open(st.session_state["selected_sample_path"])
            st.image(image_to_process, caption=f"Selected Sample: {Path(st.session_state['selected_sample_path']).parent.name} ({Path(st.session_state['selected_sample_path']).name})", use_container_width=True)
        else:
            st.markdown(f"""
            <div style="border: 2px dashed {COLOR_BORDER}; border-radius: 8px; padding: 60px 20px; text-align: center; color: {COLOR_SECONDARY_TEXT};">
                <span style="font-size: 40px;">📷</span><br>
                <div style="margin-top: 10px; font-size: 14px;">Image preview will appear here upon selection.</div>
            </div>
            """, unsafe_allow_html=True)

    if image_to_process is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        classify_clicked = st.button("🔍 Classify Waste Item", use_container_width=True)

        # Trigger on button click or auto-classify on new selection
        image_key = getattr(uploaded_file, "name", "") or st.session_state.get("selected_sample_path", "image_input")
        if classify_clicked or ("classified_result" not in st.session_state or st.session_state.get("current_image_key") != image_key):
            with st.spinner("Analyzing image via Convolutional Neural Network..."):
                try:
                    result = predict_waste(image_to_process)
                    st.session_state["classified_result"] = result
                    st.session_state["current_image_key"] = image_key
                    st.session_state.pop("classification_error", None)
                except Exception as ex:
                    import traceback
                    tb = traceback.format_exc()
                    st.session_state["classification_error"] = str(ex)
                    st.session_state["classification_tb"] = tb
                    st.session_state.pop("classified_result", None)

        if "classification_error" in st.session_state:
            st.error(f"Inference error encountered: {st.session_state['classification_error']}")
            with st.expander("Technical Traceback (For Debugging)"):
                st.code(st.session_state.get("classification_tb", "No traceback available"))

        elif "classified_result" in st.session_state:
            result = st.session_state["classified_result"]
            cat = result["predicted_category"]
            conf = result["confidence_percentage"]
            rec = result["recommendation"]
            is_uncertain = result["is_uncertain"]
            is_identified = result.get("is_identified", True)
            error_msg = result.get("error_message")

            st.markdown("---")
            st.markdown("### Classification Result")

            if not is_identified or cat == "Cannot Identify Item":
                st.error("❌ **Item Cannot Be Identified / Not a Recognizable Waste Item**")
                
                if error_msg:
                    st.warning(f"⚠️ **Detection Alert:** {error_msg}")

                c_info1, c_info2 = st.columns([1, 1], gap="medium")
                with c_info1:
                    st.markdown(f"""
                    <div class="env-card" style="border-left: 4px solid #EF4444;">
                        <h4 style="margin-top:0; color: #EF4444; font-size: 16px;">🔍 Why did this happen?</h4>
                        <ul style="font-size: 13px; color: #CBD5E1; padding-left: 18px; line-height: 1.6; margin-bottom: 0;">
                            <li><b>Digital Screenshot / UI:</b> Computer displays, web browsers, app interfaces, and text documents do not have physical material textures.</li>
                            <li><b>Non-Waste / Out-of-Distribution:</b> The image does not match any of the 6 supported physical waste categories.</li>
                            <li><b>Image Quality / Lighting:</b> Blank, completely dark, or overexposed images cannot be reliably analyzed.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                with c_info2:
                    st.markdown(f"""
                    <div class="env-card" style="border-left: 4px solid #10B981;">
                        <h4 style="margin-top:0; color: #34D399; font-size: 16px;">💡 How to get accurate classification</h4>
                        <ul style="font-size: 13px; color: #CBD5E1; padding-left: 18px; line-height: 1.6; margin-bottom: 0;">
                            <li>Upload a direct camera photo of a physical discarded object (packaging, bottle, can, organic scraps).</li>
                            <li>Place the object on a neutral background with clear, even lighting.</li>
                            <li>Avoid computer screenshots, phone screen captures, or digital graphics.</li>
                            <li>You can also test instantly using the <b>pre-loaded sample buttons</b> on the left!</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                st.info("ℹ️ **Supported Physical Waste Streams:** Cardboard • Glass • Metal • Organic Waste • Paper • Plastic")

            else:
                r_col1, r_col2 = st.columns([1, 2], gap="large")

                with r_col1:
                    st.subheader(f"{rec['icon']} {cat}")
                    st.caption("Predicted Waste Category")
                    st.metric(label="Model Confidence Score", value=f"{conf:.1f}%")

                    if is_uncertain:
                        st.warning(
                            f"⚠️ **Low Confidence Advisory:** The model returned {conf:.1f}%, which is below the "
                            f"{int(CONFIDENCE_THRESHOLD * 100)}% certainty threshold. Please verify manual sorting instructions."
                        )

                with r_col2:
                    st.markdown("#### ♻️ Disposal & Recycling Recommendation")
                    st.write(rec['disposal_instructions'])
                    
                    st.markdown("##### Preparation Steps:")
                    for p in rec['preparation_instructions']:
                        st.markdown(f"- {p}")
                    
                    st.markdown("##### Recycling & Recovery Information:")
                    st.write(rec['recycling_guidance'])
                    
                    st.info(f"**Safety & Handling Note:** {rec['safety_precautions']}")

                # Softmax Distribution Breakdown
                with st.expander("View Full Softmax Class Probability Distribution"):
                    st.caption("Exact Softmax output vector generated by final Dense layer:")
                    probs = result["class_probabilities"]
                    prob_cols = st.columns(6)
                    for i, (k, v) in enumerate(probs.items()):
                        prob_cols[i].metric(label=k, value=f"{v*100:.1f}%")


# PAGE 3: RECYCLING GUIDE
elif page == "Recycling Guide":
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <h2 class="brand-title" style="font-size: 22px;">Comprehensive Waste Segregation Guide</h2>
            <p class="brand-subtitle">Standardized handling protocols for the six major domestic recyclable and organic streams.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([f"{RECOMMENDATIONS[c]['icon']} {c}" for c in TARGET_CLASSES])

    for i, cname in enumerate(TARGET_CLASSES):
        data = RECOMMENDATIONS[cname]
        with tabs[i]:
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.subheader(f"{data['icon']} {data['display_name']}")
                st.write(data['short_description'])
                
                st.markdown("##### Common Examples")
                for ex in data['examples']:
                    st.markdown(f"- {ex}")
                
                st.markdown("##### Key Preparation Instructions")
                for idx, prep in enumerate(data['preparation_instructions'], 1):
                    st.markdown(f"**{idx}.** {prep}")

            with c2:
                st.markdown("##### 📌 Disposal Protocol")
                st.write(data['disposal_instructions'])
                
                st.markdown("##### 🔄 Recycling & Material Recovery")
                st.write(data['recycling_guidance'])
                
                st.warning(f"**Safety Precautions:** {data['safety_precautions']}")


# PAGE 4: MODEL PERFORMANCE
elif page == "Model Performance":
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <h2 class="brand-title" style="font-size: 22px;">Model Evaluation & Experimental Metrics</h2>
            <p class="brand-subtitle">Empirical test performance calculated from genuine test set inferences.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not METRICS_PATH.exists():
        st.info("Evaluation metrics file (`results/metrics.json`) not yet found. Please run `python evaluate_model.py` to evaluate.")
    else:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # High-level Summary Metrics Cards
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{metrics['overall_test_accuracy']*100:.2f}%</div>
            <div class="metric-label">Overall Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

        m_c2.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{metrics['macro_precision']*100:.2f}%</div>
            <div class="metric-label">Macro Precision</div>
        </div>
        """, unsafe_allow_html=True)

        m_c3.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{metrics['macro_recall']*100:.2f}%</div>
            <div class="metric-label">Macro Recall</div>
        </div>
        """, unsafe_allow_html=True)

        m_c4.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{metrics['macro_f1_score']*100:.2f}%</div>
            <div class="metric-label">Macro F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Class-wise Breakdown Table
        st.markdown("### Per-Class Classification Report")
        class_table_data = []
        for cname in TARGET_CLASSES:
            c_info = metrics["class_metrics"].get(cname, {})
            class_table_data.append({
                "Category": cname,
                "Precision": f"{c_info.get('precision', 0.0)*100:.1f}%",
                "Recall": f"{c_info.get('recall', 0.0)*100:.1f}%",
                "F1-Score": f"{c_info.get('f1_score', 0.0)*100:.1f}%",
                "Support (Test Count)": c_info.get('support', 0)
            })
        st.table(class_table_data)

        # Confusion Matrix and Convergence Graphs
        st.markdown("### Experimental Visualizations")
        vis_c1, vis_c2 = st.columns(2)

        with vis_c1:
            st.markdown("#### Test Confusion Matrix")
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on untouched test set", use_container_width=True)
            else:
                st.info("Confusion matrix graph not available.")

        with vis_c2:
            st.markdown("#### Training Accuracy Progression")
            if ACCURACY_GRAPH_PATH.exists():
                st.image(str(ACCURACY_GRAPH_PATH), caption="Training vs Validation Accuracy Curves", use_container_width=True)
            else:
                st.info("Accuracy graph not available.")

        st.markdown("#### Training Loss Convergence")
        if LOSS_GRAPH_PATH.exists():
            st.image(str(LOSS_GRAPH_PATH), caption="Sparse Categorical Crossentropy Loss over Epochs", use_container_width=True)

        st.caption(f"Evaluation conducted on {metrics.get('test_dataset_size', 0)} untouched test samples. Timestamp: {metrics.get('evaluation_timestamp', 'N/A')}")


# PAGE 5: ABOUT THE PROJECT & SDG
elif page == "About Project & SDG":
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <h2 class="brand-title" style="font-size: 22px;">Academic Project Overview & SDG Alignment</h2>
            <p class="brand-subtitle">Deep Learning Mini Project — Sustainable Waste Management Systems</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_info1, c_info2 = st.columns(2, gap="large")

    with c_info1:
        st.markdown("""
        ### Problem Statement
        Municipal waste generation exceeds 2 billion tonnes globally each year. Failure to sort recyclables 
        and biodegradable matter at the household level leads to contamination of recyclable streams, 
        increased landfill leachate, and high sorting costs. 

        ### Proposed Methodology
        1. **End-to-End Deep Learning**: Supervised training of a lightweight 3-block Convolutional Neural Network (CNN).
        2. **Standard Preprocessing**: Fixed 128×128 pixel RGB input scaling and training-only data augmentation.
        3. **Deterministic Guidance**: Rule-based disposal instructions linked to Softmax predictions to prevent LLM hallucinations.
        
        ### Architecture Specification
        * **Input Layer**: 128 × 128 × 3 RGB Tensor
        * **Block 1**: Conv2D(32, 3×3, ReLU) + MaxPooling2D(2×2)
        * **Block 2**: Conv2D(64, 3×3, ReLU) + MaxPooling2D(2×2)
        * **Block 3**: Conv2D(128, 3×3, ReLU) + MaxPooling2D(2×2)
        * **Head**: Flatten -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(6, Softmax)
        """)

    with c_info2:
        st.markdown(f"""
        ### UN Sustainable Development Goals (SDG) Alignment
        
        <div class="env-card" style="border-left: 4px solid #D97706;">
            <b>Primary Goal: SDG 12 — Responsible Consumption and Production</b><br>
            <span style="font-size: 13px; color: {COLOR_SECONDARY_TEXT};">
                Encourages reduction of waste generation through source segregation, prevention, reduction, recycling, and reuse (Target 12.5).
            </span>
        </div>
        
        <div class="env-card" style="border-left: 4px solid #2563EB;">
            <b>Supporting Goal: SDG 11 — Sustainable Cities and Communities</b><br>
            <span style="font-size: 13px; color: {COLOR_SECONDARY_TEXT};">
                Supports municipal solid waste management reduction and cleaner urban resource recovery (Target 11.6).
            </span>
        </div>
        
        <div class="env-card" style="border-left: 4px solid #16A34A;">
            <b>Supporting Goal: SDG 13 — Climate Action</b><br>
            <span style="font-size: 13px; color: {COLOR_SECONDARY_TEXT};">
                Diverting organic waste from anaerobic landfill environments lowers atmospheric methane emissions.
            </span>
        </div>
        
        <div class="env-card" style="border-left: 4px solid #0284C7;">
            <b>Supporting Goal: SDG 14 — Life Below Water</b><br>
            <span style="font-size: 13px; color: {COLOR_SECONDARY_TEXT};">
                Proper collection and recycling of post-consumer plastics mitigates marine litter and microplastic contamination.
            </span>
        </div>
        """, unsafe_allow_html=True)
