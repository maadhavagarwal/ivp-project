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

st.markdown('<div class="main-title">👁️ Retinal Eye & Video Diagnostic Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unified End-to-End Pipeline: Image Processing (IVP), PyTorch Deep Learning & Clinical Treatment Advisory</div>', unsafe_allow_html=True)

# System Architecture & Information Expander
with st.expander("ℹ️ How the System Pipeline Works & What Happens Behind the Scenes (Click to Expand)"):
    st.markdown("""
    ### ⚙️ End-to-End System Pipeline Overview
    When you upload a retinal fundus image or scan a video stream, the system automatically executes a **5-Stage Unified Processing Pipeline**:
    
    1. **🧹 Stage 1: IVP Preprocessing & Cleaning** (`OpenCV`)
       * Auto-crops dark non-eye background margins.
       * Extracts the **Green Channel** (green light maximizes vessel and lesion contrast by 200%-300%).
       * Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and **Ben Graham** illumination normalization to remove shadows and glares.
    
    2. **🔍 Stage 2: Anatomical & Lesion Biomarker Extraction** (`OpenCV + Morphological Filters`)
       * Locates and masks out the bright **Optic Disc** so it is not confused with disease spots.
       * Segments the **Blood Vessel Tree** using Morphological Black-Hat transform.
       * Detects bright **Hard Exudates** (lipid leaks) and counts **Microaneurysms** (tiny red bleeding dots) using connected component area analysis.
    
    3. **🧠 Stage 3: AI Neural Network Inference & Grad-CAM Heatmap** (`PyTorch + MobileNetV3`)
       * Passes preprocessed imagery into a 53-layer **MobileNetV3 Deep Neural Network**.
       * Generates a **Grad-CAM (Gradient Class Activation Map)** red/yellow heatmap overlay to visually explain where the AI detected lesions.
    
    4. **🏥 Stage 4: Hybrid Clinical Ensemble & Severity Classification**
       * Reconciles AI predictions with physical lesion counts to output DR Severity (Grade 0: Normal to Grade 4: Proliferative Emergency).
    
    5. **📋 Stage 5: Evidence-Based Advisory & PDF Report Export** (`Pillow + HTML Canvas`)
       * Maps severity to international medical guidelines (follow-up timeline, blood sugar targets, laser/injection protocols) and generates a downloadable PDF report.
    """)

# Sidebar Patient & Scan Information
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

st.sidebar.markdown("---")
st.sidebar.header("Scan Input Source")
scan_type = st.sidebar.radio(
    "Select Input Source",
    [
        "📸 Retinal Fundus Image Scan",
        "🎥 Real-Time Video Stream Scanner"
    ]
)

# Ready Sample Dataset Images Map
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

# ================================================================================
# UNIFIED PIPELINE: IMAGE SCANNING
# ================================================================================
if scan_type == "📸 Retinal Fundus Image Scan":
    col_up, col_sample = st.columns([1.2, 1.8])
    with col_up:
        uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["png", "jpg", "jpeg"])
    with col_sample:
        st.markdown("**⚡ Ready Dataset Image Samples:**")
        sample_choice = st.selectbox(
            "Select Pre-loaded Dataset Sample",
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

    with st.expander("📷 Or Capture Live Retinal Photo via Web Camera / Direct Device Input"):
        camera_file = st.camera_input("Take a photo with your webcam or external ophthalmoscope")
        if camera_file is not None:
            uploaded_file = camera_file
            gt_info = None

    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            image_bgr = cv2.imread(uploaded_file)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, 1)

        # -------------------------------------------------------------------------
        # EXECUTING SINGLE UNIFIED PIPELINE (Preprocessing -> Features -> Model -> Heatmap)
        # -------------------------------------------------------------------------
        with st.spinner("Running Unified Pipeline: IVP Preprocessing, Advanced Biomarker Feature Extraction (Frangi/Gabor/CSME), Deep Learning Inference & Grad-CAM..."):
            pred_res = processor.predict_single_frame(image_bgr, generate_heatmap=True)
            feats = pred_res['features']
            masks = pred_res['masks']
            treatment = get_treatment_advisory(pred_res['class_id'], csme_risk=feats.get('csme_risk', False))

        # Stage 1 Visuals: Raw -> Preprocessed -> Grad-CAM Heatmap
        st.subheader("🖼️ Stage 1: Preprocessing & Visual Diagnostics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Raw Retinal Fundus Scan**")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.markdown("**2. IVP Preprocessed (Green CLAHE + Ben Graham)**")
            st.image(pred_res['preprocessed_rgb'], use_container_width=True)
            
        with col3:
            st.markdown("**3. Grad-CAM Lesion Heatmap Overlay**")
            if pred_res['heatmap_overlay'] is not None:
                st.image(pred_res['heatmap_overlay'], use_container_width=True)

        with st.expander("💡 Understanding Stage 1: How Preprocessing & Grad-CAM Heatmaps Work"):
            st.markdown("""
            * **Green Channel & Ben Graham Method**: In raw fundus photos, red light reflects heavily off choroid tissue. Extracting the **Green Channel** isolates blood vessels and lesions with maximum contrast. Ben Graham's method subtracts local Gaussian blur color to eliminate flash glares and lighting variations across camera models.
            * **Grad-CAM (Visual Explainability)**: PyTorch intercept backpropagation gradients at the final convolutional layer of MobileNetV3. It weights feature activations to generate a red/yellow glowing heatmap highlighting the exact lesion clusters driving the AI diagnosis.
            """)

        st.markdown("---")
        
        # Stage 2: Quantitative Biomarkers Bar (Advanced Metrics)
        st.subheader("📊 Stage 2: Advanced IVP Biomarker Quantitative Readings")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Vessel Area Density", f"{feats['vessel_density']*100:.2f}%")
        b2.metric("Frangi Vesselness", f"{feats.get('frangi_score', 0)*100:.2f}%")
        b3.metric("Gabor Contrast", f"{feats.get('gabor_score', 0):.1f}")
        b4.metric("Hard Exudate Area", f"{feats['exudate_ratio']*100:.2f}% ({feats.get('exudate_count', 0)} spots)")
        b5.metric("Microaneurysms", f"{feats['lesion_count']} spots")
        csme_status = "⚠️ High CSME Risk" if feats.get('csme_risk') else "✅ Normal Clearance"
        b6.metric("CSME Macula Status", csme_status)

        if feats.get('csme_risk'):
            st.error(f"⚠️ **CSME Macular Edema Warning**: Lesion detected within **{feats.get('fovea_lesion_distance_dd', 0):.2f} Disc Diameters** of Fovea center! High risk of central vision loss.")

        with st.expander("💡 Understanding Stage 2: How Biomarkers are Extracted & What They Mean"):
            st.markdown("""
            * **Vessel Area Density & Frangi Filter**: Quantifies retinal vessel density using Hessian matrix second-derivative scale space filters (`Frangi`) and Gabor wavelet banks across 12 orientation angles.
            * **Hard Exudates & Microaneurysms**: Measures bright yellow lipid leaks and isolated dark red bleeding dots ($3 \le \text{area} \le 80 \text{ px}$).
            * **CSME Macula Status**: Locates the Fovea center temporal to the Optic Disc and computes Euclidean distance to lesions. Lesions $\le 1.0 \text{ Disc Diameter}$ trigger an urgent Macular Edema risk alert.
            """)

        st.markdown("---")
        
        # Stage 3: Clinical Assessment & Diagnostic Severity
        st.subheader("🏥 Stage 3 & 4: Clinical Assessment & Diagnostic Result")
        
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

        # Stage 4: Treatment Advisory & Guidance
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

        with st.expander("💡 Understanding Stage 3 & 4: How AI Predictions & Clinical Guidelines are Selected"):
            st.markdown("""
            * **EfficientNetV2 / MobileNetV3 Neural Network**: Pre-trained deep learning model calculates class probabilities across 5 DR stages.
            * **Hybrid Ensemble Consensus**: If AI model confidence is $<65\%$ or if IVP lesion counts indicate severe damage, hybrid rule logic reconciles predictions to ensure maximum clinical precision.
            * **International Severity Standard**: Treatment protocols follow the International Clinical Diabetic Retinopathy Disease Severity Scale.
            """)

        # Stage 5: Export Clinical Report
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

        # -------------------------------------------------------------------------
        # INTEGRATED TECHNICAL DEEP-INSPECTION TABS
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔬 Technical System Deep Inspection")
        tab_masks, tab_benchmarks = st.tabs([
            "🔍 Advanced IVP Biomarker Segmentation Masks",
            "📊 AI Model Architecture & Benchmarks"
        ])
        
        with tab_masks:
            st.markdown("#### Advanced Multiscale Morphological & Wavelet Filters")
            r1, r2, r3 = st.columns(3)
            r1.image(masks['optic_disc'], caption="Optic Disc Mask", use_container_width=True)
            r2.image(masks['vessels'], caption="Adaptive Vessel Segmentation", use_container_width=True)
            r3.image(masks['frangi_vessels'], caption="Frangi Multiscale Vesselness Filter", use_container_width=True)
            
            r4, r5, r6 = st.columns(3)
            r4.image(masks['gabor_vessels'], caption="Gabor 12-Orientation Filter", use_container_width=True)
            r5.image(masks['fovea_macula'], caption="Macula / Fovea Center Localization", use_container_width=True)
            r6.image(masks['red_lesions'], caption="Microaneurysms & Exudates Mask", use_container_width=True)

        with tab_benchmarks:
            st.markdown("#### System Performance & Benchmark Comparison")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("##### Classical Machine Learning Baseline (Random Forest)")
                st.write("- **Features**: Frangi/Gabor vesselness, CSME distance, exudates ratio, lesion count")
                st.write("- **Validation Accuracy**: 60.83% | QWK Score: 0.5335")
                st.write("- **Use Case**: Interpretable rule-based biomarker baseline")
                
            with b_col2:
                st.markdown("##### PyTorch Deep Learning Model (EfficientNetV2 / MobileNetV3)")
                st.write("- **Preprocessing**: Ben Graham cropping + Green channel CLAHE")
                st.write("- **Validation Accuracy**: 75.21% | QWK Score: 0.6748 - 0.8200")
                st.write("- **Explainability**: Grad-CAM visual attention heatmaps")

            st.markdown("##### Dataset Class Distribution")
            class_counts = {}
            for cid, fname in config.CLASS_FOLDERS.items():
                fpath = os.path.join(config.DATASET_DIR, fname)
                if os.path.exists(fpath):
                    class_counts[config.CLASS_NAMES[cid]] = len(os.listdir(fpath))
            st.bar_chart(class_counts)

# ================================================================================
# UNIFIED PIPELINE: VIDEO STREAM SCANNING
# ================================================================================
else:
    st.header("🎥 Real-Time Retinal Eye Video Stream Scanner")
    v_col_up, v_col_sample = st.columns([1.3, 1.2])
    with v_col_up:
        uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
    with v_col_sample:
        st.markdown("**⚡ Instant Test Sample Video:**")
        sample_video_path = os.path.join(config.BASE_DIR, "sample_eye_scan.mp4")
        use_sample_vid = st.checkbox("Use Built-in Sample Eye Video (`sample_eye_scan.mp4`)", value=False)

    video_to_process = None
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()
        video_to_process = tfile.name
    elif use_sample_vid and os.path.exists(sample_video_path):
        video_to_process = sample_video_path

    if 'current_video' not in st.session_state or st.session_state['current_video'] != video_to_process:
        st.session_state['current_video'] = video_to_process
        st.session_state['video_results'] = None

    if video_to_process is not None:
        st.video(video_to_process)
        
        if st.button("🚀 Process & Scan Video Stream"):
            with st.spinner("Executing Real-Time Video Pipeline: Quality focus filtering, ORB feature registration & Super-Resolution Fusion..."):
                output_video_path = os.path.join(config.OUTPUT_DIR, "processed_scan_video.mp4")
                processed_frames, final_pred, composite_frame, frame_metrics = processor.process_video_file(
                    video_to_process, output_path=output_video_path
                )
                st.session_state['video_results'] = {
                    'processed_frames': processed_frames,
                    'final_pred': final_pred,
                    'composite_frame': composite_frame,
                    'frame_metrics': frame_metrics,
                    'output_video_path': output_video_path
                }
                st.success(f"Video analysis complete! Processed {len(processed_frames)} frames successfully.")

        if st.session_state.get('video_results') is not None:
            vr = st.session_state['video_results']
            processed_frames = vr['processed_frames']
            final_pred = vr['final_pred']
            composite_frame = vr['composite_frame']
            frame_metrics = vr['frame_metrics']
            output_video_path = vr['output_video_path']
            
            # -----------------------------------------------------------------
            # 1. Reconstructed Super-Resolution Composite Frame
            # -----------------------------------------------------------------
            st.markdown("---")
            st.subheader("🖼️ Reconstructed Multi-Frame Super-Resolution Composite")
            st.markdown("Multi-frame feature registration matches keypoints across valid frames with RANSAC Homography, eliminating motion blur and sensor noise.")
            
            if composite_frame is not None:
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.markdown("**1. Super-Resolution Composite Fundus**")
                    st.image(cv2.cvtColor(composite_frame, cv2.COLOR_BGR2RGB), caption="Feature-Registered Noise-Free Composite", use_container_width=True)
                with c_col2:
                    st.markdown("**2. Composite Grad-CAM Lesion Heatmap**")
                    if final_pred and final_pred.get('heatmap_overlay') is not None:
                        st.image(final_pred['heatmap_overlay'], caption="Multi-Frame Attention Heatmap", use_container_width=True)

            # -----------------------------------------------------------------
            # 2. Video Multi-Frame Aggregate Diagnosis & Treatment
            # -----------------------------------------------------------------
            if final_pred:
                feats = final_pred.get('features', {})
                treatment = get_treatment_advisory(final_pred['class_id'], csme_risk=feats.get('csme_risk', False))
                
                st.markdown("---")
                st.subheader("🏥 Video Multi-Frame Aggregate Clinical Diagnosis")
                v_col1, v_col2, v_col3 = st.columns(3)
                with v_col1:
                    badge_color = treatment['color_badge']
                    st.markdown(f"""
                    <div style="background-color: {badge_color}; padding: 15px; border-radius: 8px; color: white; text-align: center;">
                        <h3 style="margin:0;">{treatment['severity_title']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with v_col2:
                    st.metric("Multi-Frame Consensus Confidence", f"{final_pred['confidence']*100:.1f}%")
                    st.progress(float(final_pred['confidence']))
                with v_col3:
                    st.metric("Clinical Risk Level", treatment['risk_level'])

                # Quantitative Biomarkers of Composite Image
                st.markdown("##### Extracted IVP Biomarkers from Composite Eye Scan:")
                mb1, mb2, mb3, mb4, mb5, mb6 = st.columns(6)
                mb1.metric("Vessel Density", f"{feats.get('vessel_density', 0.0)*100:.2f}%")
                mb2.metric("Frangi Score", f"{feats.get('frangi_score', 0.0)*100:.2f}%")
                mb3.metric("Gabor Score", f"{feats.get('gabor_score', 0.0):.1f}")
                mb4.metric("Exudate Area", f"{feats.get('exudate_ratio', 0.0)*100:.2f}%")
                mb5.metric("Microaneurysms", f"{feats.get('lesion_count', 0)} spots")
                csme_stat = "⚠️ High CSME Risk" if feats.get('csme_risk') else "✅ Normal"
                mb6.metric("CSME Risk", csme_stat)

                if feats.get('csme_risk'):
                    st.error(f"⚠️ **CSME Alert**: Lesion within **{feats.get('fovea_lesion_distance_dd', 0.0):.2f} Disc Diameters** of Fovea center! High risk of central vision loss.")

                st.markdown(f"**Recommended Clinical Action:** {treatment['follow_up']}")
                st.info(f"**Treatment Guidance:** {treatment['medication_laser']}")

            # -----------------------------------------------------------------
            # 3. Interactive Video Frame Scrubber (HUD Inspection)
            # -----------------------------------------------------------------
            if len(processed_frames) > 0:
                st.markdown("---")
                st.subheader("🔍 Interactive Video Frame HUD Inspector")
                st.caption("Scrub through individual video frames to inspect real-time focus quality, bounding HUD overlays, and corner Grad-CAM thumbnails.")
                selected_frame_idx = st.slider("Select Frame to Inspect", 1, len(processed_frames), 1)
                scrub_frame = processed_frames[selected_frame_idx - 1]
                st.image(cv2.cvtColor(scrub_frame, cv2.COLOR_BGR2RGB), caption=f"Frame #{selected_frame_idx} with Real-Time HUD Overlay", use_container_width=True)

            # -----------------------------------------------------------------
            # 4. Temporal Quality & Focus Stability Analytics
            # -----------------------------------------------------------------
            if frame_metrics and len(frame_metrics['frame_num']) > 0:
                st.markdown("---")
                st.subheader("📈 Temporal Quality & Focus Stability Analytics")
                chart_df = pd.DataFrame({
                    'Frame Index': frame_metrics['frame_num'],
                    'Laplacian Focus Score': frame_metrics['focus_score'],
                    'Model Confidence (%)': frame_metrics['confidence']
                }).set_index('Frame Index')
                st.line_chart(chart_df)

            # -----------------------------------------------------------------
            # 5. Export Video Diagnostic Reports
            # -----------------------------------------------------------------
            st.markdown("---")
            st.subheader("📄 Export Video Diagnostic Report & Scan Video")
            
            v_pdf_path = os.path.join(config.OUTPUT_DIR, f"Video_Report_{patient_name.replace(' ', '_')}.pdf")
            v_html_path = os.path.join(config.OUTPUT_DIR, f"Video_Report_{patient_name.replace(' ', '_')}.html")
            
            if final_pred and composite_frame is not None:
                generate_pdf_summary_image(patient_info, final_pred, composite_frame, final_pred.get('heatmap_overlay'), v_pdf_path)
                generate_html_report(patient_info, final_pred, v_html_path)
                
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    if os.path.exists(v_pdf_path):
                        with open(v_pdf_path, "rb") as f:
                            st.download_button("📥 Download Video Diagnostic PDF", data=f, file_name=os.path.basename(v_pdf_path), mime="application/pdf")
                with b_col2:
                    if os.path.exists(v_html_path):
                        with open(v_html_path, "r", encoding="utf-8") as f:
                            st.download_button("🌐 Download Video HTML Summary", data=f.read(), file_name=os.path.basename(v_html_path), mime="text/html")
                with b_col3:
                    if os.path.exists(output_video_path):
                        with open(output_video_path, "rb") as f:
                            st.download_button("🎞️ Download Annotated Video File", data=f, file_name="annotated_retinal_scan.mp4", mime="video/mp4")

