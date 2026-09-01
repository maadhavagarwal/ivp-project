import streamlit as st
import cv2
import numpy as np
import os
import torch
import pandas as pd
from PIL import Image
import tempfile

from src import config
from src.ivp_preprocessing import full_ivp_pipeline
from src.feature_extraction import extract_all_ivp_features
from src.video_processor import VideoProcessor
from src.treatment_advisor import get_treatment_advisory
from src.pdf_generator import generate_html_report, generate_pdf_summary_image

# Page Setup
st.set_page_config(
    page_title="Diabetic Retinopathy Eye & Video Scanner",
    page_icon="👁️",
    layout="wide"
)

# Initialize VideoProcessor in session state
@st.cache_resource
def load_video_processor():
    return VideoProcessor()

processor = load_video_processor()

# Custom CSS Styling
st.markdown("""
<style>
    .main-title { font-size: 32px; font-weight: bold; color: #1d3557; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #457b9d; margin-bottom: 25px; }
    .badge-card { padding: 15px; border-radius: 10px; color: white; font-weight: bold; font-size: 18px; margin-bottom: 15px; }
    .stButton>button { background-color: #3a86ef; color: white; border-radius: 8px; font-weight: bold; }
    .metric-card { background-color: #f8f9fa; padding: 10px; border-radius: 6px; border-left: 4px solid #3a86ef; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">👁️ Retinal Eye & Video Scanner for Diabetic Retinopathy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Image Processing (IVP), PyTorch Deep Learning & Clinical Treatment Advisory System</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/eye-checked.png", width=70)
st.sidebar.header("Patient & Scan Metadata")
patient_name = st.sidebar.text_input("Patient Name", "John Doe")
patient_age = st.sidebar.text_input("Age", "52")
patient_sex = st.sidebar.selectbox("Sex", ["Male", "Female", "Other"])
examined_eye = st.sidebar.selectbox("Examined Eye", ["Right Eye (OD)", "Left Eye (OS)"])

patient_info = {
    'name': patient_name,
    'age': patient_age,
    'sex': patient_sex,
    'eye': examined_eye
}

app_mode = st.sidebar.radio(
    "Select System Phase / Mode",
    [
        "Phase 1 & 5: Retinal Image Scan & Diagnosis",
        "Phase 4: Real-Time Video Scanner",
        "Phase 2: IVP Biomarker & Feature Inspection",
        "Phase 3: Model Performance & Benchmark"
    ]
)

# Ready Sample Dataset Images Map (2 distinct dataset images per severity class)
READY_SAMPLES = {
    "Custom Upload": None,
    "Grade 0: No DR - Sample Image 1 (002c21358ce6)": ("No_DR", "002c21358ce6.png", 0, "002c21358ce6"),
    "Grade 0: No DR - Sample Image 2 (005b95c28852)": ("No_DR", "005b95c28852.png", 0, "005b95c28852"),
    "Grade 1: Mild DR - Sample Image 1 (0024cdab0c1e)": ("Mild", "0024cdab0c1e.png", 1, "0024cdab0c1e"),
    "Grade 1: Mild DR - Sample Image 2 (00cb6555d108)": ("Mild", "00cb6555d108.png", 1, "00cb6555d108"),
    "Grade 2: Moderate DR - Sample Image 1 (000c1434d8d7)": ("Moderate", "000c1434d8d7.png", 2, "000c1434d8d7"),
    "Grade 2: Moderate DR - Sample Image 2 (00a8624548a9)": ("Moderate", "00a8624548a9.png", 2, "00a8624548a9"),
    "Grade 3: Severe DR - Sample Image 1 (0104b032c141)": ("Severe", "0104b032c141.png", 3, "0104b032c141"),
    "Grade 3: Severe DR - Sample Image 2 (03c85870824c)": ("Severe", "03c85870824c.png", 3, "03c85870824c"),
    "Grade 4: Proliferative DR - Sample Image 1 (001639a390f0)": ("Proliferate_DR", "001639a390f0.png", 4, "001639a390f0"),
    "Grade 4: Proliferative DR - Sample Image 2 (0083ee8054ee)": ("Proliferate_DR", "0083ee8054ee.png", 4, "0083ee8054ee")
}

# --------------------------------------------------------------------------------
# Mode 1: Single Image Scan & Treatment Recommendation
# --------------------------------------------------------------------------------
if app_mode == "Phase 1 & 5: Retinal Image Scan & Diagnosis":
    st.header("Phase 1 & Phase 5: Fundus Image Scanning & Clinical Advisory")
    
    col_up, col_sample = st.columns([1.2, 1.8])
    with col_up:
        uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["png", "jpg", "jpeg"])
    with col_sample:
        st.markdown("**⚡ Ready Dataset Image Samples (Select 1 or 2 images per class):**")
        sample_choice = st.selectbox(
            "Select Ready Dataset Image",
            list(READY_SAMPLES.keys())
        )
        
        gt_info = None
        if sample_choice != "Custom Upload":
            folder, fname, gt_class_id, img_id = READY_SAMPLES[sample_choice]
            s_path = os.path.join(config.DATASET_DIR, folder, fname)
            if os.path.exists(s_path):
                uploaded_file = s_path
                gt_info = {
                    'id': img_id,
                    'gt_class_id': gt_class_id,
                    'gt_class_name': config.CLASS_NAMES[gt_class_id]
                }

    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            image_bgr = cv2.imread(uploaded_file)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, 1)

        # Run Prediction and IVP Preprocessing
        pred_res = processor.predict_single_frame(image_bgr, generate_heatmap=True)
        treatment = get_treatment_advisory(pred_res['class_id'])
        
        # Extract Real Quantitative IVP Feature Metrics directly from prediction engine
        feats = pred_res['features']
        masks = pred_res['masks']
        
        # Display side-by-side images
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Raw Fundus Scan")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.subheader("2. Phase 1 IVP Preprocessed")
            st.image(pred_res['preprocessed_rgb'], use_container_width=True)
            
        with col3:
            st.subheader("3. Grad-CAM Lesion Heatmap")
            if pred_res['heatmap_overlay'] is not None:
                st.image(pred_res['heatmap_overlay'], use_container_width=True)

        st.markdown("---")
        
        # Real Extracted Biomarker Quantitative Data Bar
        st.subheader("📊 Extracted IVP Biomarker Quantitative Readings")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Vessel Area Density", f"{feats['vessel_density']*100:.2f}%")
        b2.metric("Hard Exudate Area Ratio", f"{feats['exudate_ratio']*100:.2f}%")
        b3.metric("Microaneurysms (Red Lesions)", f"{feats['lesion_count']} spots")
        b4.metric("Green Channel Contrast", f"{feats['clahe_contrast']:.1f}")

        st.markdown("---")
        
        # Diagnostic Severity & Treatment Advisory Section
        st.subheader("Clinical Assessment & Diagnostic Result")
        
        if gt_info is not None:
            st.info(f"📋 **Dataset Record (`train.csv`) ID:** `{gt_info['id']}` | **Ground Truth Label:** Grade {gt_info['gt_class_id']} ({gt_info['gt_class_name']})")
        
        c_badge, c_conf, c_risk = st.columns(3)
        with c_badge:
            badge_color = treatment['color_badge']
            st.markdown(f"""
            <div style="background-color: {badge_color}; padding: 15px; border-radius: 8px; color: white; text-align: center;">
                <h3 style="margin:0;">{treatment['severity_title']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        with c_conf:
            st.metric("Model Prediction Confidence", f"{pred_res['confidence']*100:.1f}%")
            st.progress(float(pred_res['confidence']))
            
        with c_risk:
            st.metric("Clinical Risk Level", treatment['risk_level'])

        # Detailed Treatment Guidelines
        st.subheader("📋 Recommended Level of Treatment Protocol")
        
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(f"**Follow-Up Schedule:** {treatment['follow_up']}")
            st.markdown(f"**Metabolic Target:** {treatment['glycemic_target']}")
            st.markdown(f"**Description:** {treatment['description']}")
            
        with t_col2:
            st.markdown("**Required Clinical Actions:**")
            for act in treatment['clinical_actions']:
                st.markdown(f"- {act}")
                
            st.warning(f"**Medication / Laser Guidance:** {treatment['medication_laser']}")

        # Report Generator Buttons
        st.markdown("---")
        st.subheader("📄 Export Clinical Diagnostic Report")
        
        pdf_out_path = os.path.join(config.OUTPUT_DIR, f"Report_{patient_name.replace(' ', '_')}.pdf")
        html_out_path = os.path.join(config.OUTPUT_DIR, f"Report_{patient_name.replace(' ', '_')}.html")
        
        generate_pdf_summary_image(patient_info, pred_res, image_bgr, pred_res['heatmap_overlay'], pdf_out_path)
        generate_html_report(patient_info, pred_res, html_out_path)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if os.path.exists(pdf_out_path):
                with open(pdf_out_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Diagnostic PDF Report",
                        data=f,
                        file_name=os.path.basename(pdf_out_path),
                        mime="application/pdf"
                    )
        with btn_col2:
            if os.path.exists(html_out_path):
                with open(html_out_path, "r") as f:
                    st.download_button(
                        label="🌐 Download HTML Diagnostic Summary",
                        data=f.read(),
                        file_name=os.path.basename(html_out_path),
                        mime="text/html"
                    )

# --------------------------------------------------------------------------------
# Mode 2: Phase 4 Video Scanner
# --------------------------------------------------------------------------------
elif app_mode == "Phase 4: Real-Time Video Scanner":
    st.header("Phase 4: Real-Time Retinal Eye Video Processing Engine")
    st.info("Upload an eye scan video file (.mp4, .avi, .mov) for multi-frame focus filtering, stability voting, and lesion tracking.")
    
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        st.video(tfile.name)
        
        if st.button("🚀 Process & Scan Video Stream"):
            with st.spinner("Analyzing video frames, checking focus quality, and computing aggregate diagnosis..."):
                output_video_path = os.path.join(config.OUTPUT_DIR, "processed_scan_video.mp4")
                processed_frames, final_pred = processor.process_video_file(
                    tfile.name, output_path=output_video_path
                )
                
                st.success(f"Video analysis complete! Processed {len(processed_frames)} frames.")
                
                if final_pred:
                    treatment = get_treatment_advisory(final_pred['class_id'])
                    
                    st.subheader("Video Multi-Frame Aggregate Diagnosis")
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.markdown(f"**Predicted Severity:** Grade {final_pred['class_id']} ({final_pred['class_name']})")
                        st.markdown(f"**Stability Voting Confidence:** {final_pred['confidence']*100:.1f}%")
                    with v_col2:
                        st.markdown(f"**Clinical Risk:** {treatment['risk_level']}")
                        st.markdown(f"**Recommended Action:** {treatment['follow_up']}")

# --------------------------------------------------------------------------------
# Mode 3: Phase 2 Feature Extraction Inspection
# --------------------------------------------------------------------------------
elif app_mode == "Phase 2: IVP Biomarker & Feature Inspection":
    st.header("Phase 2: Image Processing (IVP) Biomarker Extraction")
    
    uploaded_file = st.file_uploader("Upload Retinal Image for Biomarker Inspection", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        sample_path = os.path.join(config.DATASET_DIR, "Mild", "0024cdab0c1e.png")
        if os.path.exists(sample_path):
            uploaded_file = sample_path
            
    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            img_bgr = cv2.imread(uploaded_file)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)

        feats, masks = extract_all_ivp_features(img_bgr)
        
        st.subheader("1. Extracted Quantitative Biomarkers")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Vessel Area Density", f"{feats['vessel_density']*100:.2f}%")
        m2.metric("Exudate Area Ratio", f"{feats['exudate_ratio']*100:.2f}%")
        m3.metric("Red Lesions Count", feats['lesion_count'])
        m4.metric("Green Channel Contrast", f"{feats['clahe_contrast']:.1f}")

        st.subheader("2. Segmentation Masks & IVP Filters")
        r1, r2, r3, r4 = st.columns(4)
        r1.image(masks['optic_disc'], caption="Optic Disc Mask", use_container_width=True)
        r2.image(masks['vessels'], caption="Vessel Tree Segmentation", use_container_width=True)
        r3.image(masks['exudates'], caption="Hard Exudates Mask", use_container_width=True)
        r4.image(masks['red_lesions'], caption="Red Lesions (Microaneurysms)", use_container_width=True)

# --------------------------------------------------------------------------------
# Mode 4: Phase 3 Model Performance
# --------------------------------------------------------------------------------
elif app_mode == "Phase 3: Model Performance & Benchmark":
    st.header("Phase 3: Model Performance & Benchmark Evaluation")
    
    st.subheader("Dataset Class Distribution")
    class_counts = {}
    for cid, fname in config.CLASS_FOLDERS.items():
        fpath = os.path.join(config.DATASET_DIR, fname)
        if os.path.exists(fpath):
            class_counts[config.CLASS_NAMES[cid]] = len(os.listdir(fpath))
            
    st.bar_chart(class_counts)
    
    st.subheader("Model Benchmark Comparison")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("### Classical Computer Vision Baseline (Random Forest)")
        st.write("- **Features**: Gabor/Frangi vessels, exudates ratio, lesion count, CLAHE contrast")
        st.write("- **Validation Accuracy**: 60.83% | QWK: 0.5335")
        st.write("- **Use Case**: Interpretable rule-based biomarker feature baseline")
        
    with b_col2:
        st.markdown("### PyTorch Deep Learning Model (MobileNetV3 / ResNet)")
        st.write("- **Preprocessing**: Ben Graham cropping + Green channel CLAHE")
        st.write("- **Training Accuracy**: 75.21% | Val Accuracy: 59.17% | QWK: 0.6748")
        st.write("- **Explainability**: Grad-CAM visual attention heatmaps")
