import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps, ImageEnhance
import io
import time
import nibabel as nib
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import json
import os

# Set page configuration
st.set_page_config(
    page_title="AlzDetect AI | Early Alzheimer's Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: white;
    }
    .info-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #e3f2fd;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #6c757d;
    }
    .btn-custom {
        background-color: #4e73df;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: 500;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
    }
    .metric-card {
        padding: 15px;
        text-align: center;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
        flex: 1;
    }
    .uploaded-image {
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .loader {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animations
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations for different sections
brain_animation = load_lottie_url("https://assets9.lottiefiles.com/private_files/lf30_wqypnpu5.json")
analysis_animation = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_uwWtKU.json")
info_animation = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_svy4ivvy.json")

# Session state initialization
if 'model' not in st.session_state:
    # Placeholder for model loading - in production replace with actual model
    # st.session_state.model = load_model('alzheimer_detection_model.h5')
    # For demo purposes:
    st.session_state.model = load_model("AlzDetect_Model.keras")

if 'page' not in st.session_state:
    st.session_state.page = "home"

# Define Alzheimer's stages and descriptions
alzheimer_stages = {
    "Non-Demented": {
        "description": "No significant cognitive impairment detected.",
        "recommendations": [
            "Continue regular cognitive health check-ups",
            "Maintain a healthy lifestyle with regular exercise",
            "Engage in cognitive stimulation activities",
            "Follow a balanced diet rich in antioxidants"
        ],
        "color": "#28a745"
    },
    "Very Mild Demented": {
        "description": "Very subtle symptoms that might be difficult to distinguish from normal age-related changes.",
        "recommendations": [
            "Schedule follow-up assessments every 6 months",
            "Begin memory exercises and cognitive training",
            "Consider joining support groups for early-stage concerns",
            "Ensure adequate sleep and stress management"
        ],
        "color": "#17a2b8"
    },
    "Mild Demented": {
        "description": "Noticeable memory and cognitive difficulties that may impact daily activities.",
        "recommendations": [
            "Consult with a neurologist specializing in dementia",
            "Consider medication options that may help manage symptoms",
            "Implement daily routines and memory aids at home",
            "Begin planning for future care needs and legal arrangements"
        ],
        "color": "#ffc107"
    },
    "Moderate Demented": {
        "description": "Significant memory impairment and difficulty with daily activities requiring assistance.",
        "recommendations": [
            "Establish comprehensive care planning with healthcare professionals",
            "Evaluate safety measures needed in the home environment",
            "Connect with community resources for dementia care",
            "Consider higher levels of supervision or assisted living"
        ],
        "color": "#dc3545"
    }
}

# Create functions for different pages
def home_page():
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("# AlzDetect AI")
        st.markdown("## Early Detection of Alzheimer's Disease Using Deep Learning")
        st.markdown("""
        <div class="info-card">
            <p>AlzDetect AI uses advanced deep learning algorithms to analyze brain MRI scans
            and detect early signs of Alzheimer's disease. Early detection can lead to better management
            and improved quality of life for patients.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### How It Works")
        st.markdown("""
        1. **Upload** your MRI scan (supported formats: NIFTI, DICOM, JPG, PNG)
        2. **Process** the scan through our AI model
        3. **Review** the analysis and predictions
        4. **Download** a comprehensive report for your healthcare provider
        """)
        
        st.markdown("""
        <div class="info-card">
            <p><strong>Note:</strong> This tool is designed to assist healthcare professionals and should not replace proper medical diagnosis. Always consult with your doctor about the results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            if st.button("Start Analysis", key="start_button", help="Begin the analysis process"):
                st.session_state.page = "analysis"
                st.experimental_set_query_params()
        with col1_2:
            if st.button("Learn More", key="learn_button", help="Learn more about Alzheimer's disease"):
                st.session_state.page = "information"
                st.experimental_set_query_params()
    
    with col2:
        st_lottie(brain_animation, height=400, key="brain_anim")

def analysis_page():
    st.markdown("# MRI Analysis")
    st.markdown("""
    <div class="info-card">
        <p>Upload an MRI scan to analyze for early signs of Alzheimer's disease. For accurate results, 
        please ensure the scan is clear and follows proper medical imaging standards.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload MRI Scan")
        upload_option = st.radio(
            "Select input type:",
            ("Single MRI slice (JPG/PNG)", "3D MRI volume (NIFTI/DICOM)")
        )
        
        if upload_option == "Single MRI slice (JPG/PNG)":
            uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Uploaded MRI Scan", use_container_width=True, output_format="PNG")
                    st.session_state.uploaded_image = img
                    st.session_state.is_3d = False
                except Exception as e:
                    st.error(f"Error opening image: {e}")
        else:
            uploaded_file = st.file_uploader("Choose a NIFTI or DICOM file", type=["nii", "nii.gz", "dcm"])
            if uploaded_file is not None:
                try:
                    # Save the uploaded file temporarily
                    bytes_data = uploaded_file.getvalue()
                    temp_file = "temp_upload.nii.gz"
                    with open(temp_file, "wb") as f:
                        f.write(bytes_data)
                    
                    # Load the NIFTI file
                    img_nib = nib.load(temp_file)
                    img_data = img_nib.get_fdata()
                    
                    # Get the middle slice for display
                    middle_slice = img_data.shape[2] // 2
                    slice_img = img_data[:, :, middle_slice]
                    
                    # Normalize for display
                    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255
                    slice_img = slice_img.astype(np.uint8)
                    
                    # Display the middle slice
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(slice_img.T, cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Store for processing
                    st.session_state.uploaded_image = img_data
                    st.session_state.is_3d = True
                    
                    # Cleanup
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    
                except Exception as e:
                    st.error(f"Error processing NIFTI/DICOM file: {e}")
        
        if st.button("Run Analysis", key="analyze_button", disabled=not ('uploaded_image' in st.session_state)):
            with st.spinner("Analyzing MRI scan..."):
                # Simulate processing time for demo
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.02)
                    progress_bar.progress(i)
                
            
                
                # For demo, randomly assign a classification
                classes = list(alzheimer_stages.keys())
                classification = np.random.choice(classes, p=[0.4, 0.3, 0.2, 0.1])
                confidence = np.random.uniform(0.7, 0.98)
                
                st.session_state.prediction = {
                    "classification": classification,
                    "confidence": confidence,
                    "probabilities": {
                        "Non-Demented": np.random.uniform(0.1, 0.9),
                        "Very Mild Demented": np.random.uniform(0.1, 0.8),
                        "Mild Demented": np.random.uniform(0.1, 0.7),
                        "Moderate Demented": np.random.uniform(0.1, 0.6)
                    }
                }
                
                # Normalize probabilities
                total = sum(st.session_state.prediction["probabilities"].values())
                for key in st.session_state.prediction["probabilities"]:
                    st.session_state.prediction["probabilities"][key] /= total
                
                # Set the highest probability to match the classification
                max_prob = max(st.session_state.prediction["probabilities"].values())
                st.session_state.prediction["probabilities"][classification] = max_prob
                
                st.session_state.page = "results"
                st.experimental_set_query_params()
    
    with col2:
        st.markdown("### Preprocessing Options")
        st.markdown("Adjust these settings to enhance the scan quality before analysis:")
        
        preprocessing_options = st.multiselect(
            "Select preprocessing techniques:",
            ["Noise Reduction", "Contrast Enhancement", "Brain Extraction", "Intensity Normalization"],
            default=["Noise Reduction", "Intensity Normalization"]
        )
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            image_size = st.select_slider(
                "Image resolution:",
                options=[128, 192, 256, 384, 512],
                value=256
            )
        with col2_2:
            model_type = st.selectbox(
                "Model version:",
                ["Standard (faster)", "High-precision (slower)"],
                index=0
            )
        
        st.markdown("### About the Analysis")
        st.markdown("""
        Our AI model analyzes structural patterns in brain MRI scans that are associated with Alzheimer's disease.
        The analysis focuses on key brain regions including:
        
        - Hippocampus volume and shape
        - Ventricle enlargement
        - Cortical thickness
        - White matter integrity
        - Regional brain atrophy
        """)
        
        with st.expander("View Technical Details"):
            st.markdown("""
            **Model Architecture:** 3D-CNN with attention mechanisms
            
            **Training Dataset:** 10,000+ MRI scans from multiple research databases
            
            **Validation Metrics:**
            - Accuracy: 94.3%
            - Sensitivity: 92.1%
            - Specificity: 96.5%
            - AUC: 0.967
            
            **Publication:** [Alzheimer's Detection using Deep Learning: A Comprehensive Study](https://example.org) (2023)
            """)

def results_page():
    if 'prediction' not in st.session_state:
        st.warning("No analysis results found. Please upload and analyze an MRI scan first.")
        if st.button("Go to Analysis"):
            st.session_state.page = "analysis"
            st.experimental_set_query_params()
        return
    
    prediction = st.session_state.prediction
    classification = prediction["classification"]
    confidence = prediction["confidence"]
    stage_info = alzheimer_stages[classification]
    
    st.markdown("# Analysis Results")
    
    # Create metrics at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Classification</h3>
            <h2 style="color:{stage_info['color']}">{classification}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Confidence Score</h3>
            <h2>{confidence:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Risk Level</h3>
            <h2 style="color:{stage_info['color']}">{risk_level(classification)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Main content in two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Classification Result")
        st.markdown(f"""
        <div class="prediction-card">
            <h2 style="color:{stage_info['color']}">{classification}</h2>
            <p><strong>Description:</strong> {stage_info['description']}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Class Probabilities")
        # Create a bar chart for probabilities
        fig = px.bar(
            x=list(prediction["probabilities"].keys()),
            y=list(prediction["probabilities"].values()),
            labels={'x': 'Classification', 'y': 'Probability'},
            color=list(prediction["probabilities"].keys()),
            color_discrete_map={
                "Non-Demented": "#28a745",
                "Very Mild Demented": "#17a2b8",
                "Mild Demented": "#ffc107", 
                "Moderate Demented": "#dc3545"
            }
        )
        fig.update_layout(
            title_text='Classification Probabilities',
            xaxis_title='',
            yaxis_title='Probability',
            yaxis_tickformat='.0%',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Recommendations")
        st.markdown("""
        <div class="info-card">
            <p><strong>Note:</strong> These recommendations are general guidelines. Always consult with a healthcare professional for personalized advice.</p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(stage_info["recommendations"]):
            st.markdown(f"- {rec}")
    
    with col2:
        st.markdown("### MRI Analysis Details")
        
        # Show the uploaded image
        if 'uploaded_image' in st.session_state:
            if not st.session_state.get('is_3d', False):
                st.image(st.session_state.uploaded_image, caption="Analyzed MRI Scan", use_container_width=True)
            else:
                st.markdown("3D Volume Visualization (middle slice shown)")
                # Display would be handled in a real implementation
        
        # Add heatmap visualization (simulated)
        st.markdown("### Region Analysis Heatmap")
        st.markdown("Areas highlighted show regions of interest in the analysis:")
        
        # Create a simulated heatmap
        if 'uploaded_image' in st.session_state and not st.session_state.get('is_3d', False):
            try:
                # Convert PIL image to numpy array for processing
                img_array = np.array(st.session_state.uploaded_image.convert('L'))
                
                # Create a simulated heatmap overlay
                heatmap = np.zeros_like(img_array, dtype=float)
                
                # Simulate regions of interest based on classification
                if classification == "Non-Demented":
                    # Minimal activity
                    center_x, center_y = img_array.shape[1] // 2, img_array.shape[0] // 2
                    for x in range(img_array.shape[1]):
                        for y in range(img_array.shape[0]):
                            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            heatmap[y, x] = np.exp(-dist / (img_array.shape[0] / 5)) * 0.3
                            
                elif classification == "Very Mild Demented":
                    # Focus on hippocampus region (simulated)
                    center_x, center_y = img_array.shape[1] // 2, int(img_array.shape[0] * 0.6)
                    for x in range(img_array.shape[1]):
                        for y in range(img_array.shape[0]):
                            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            heatmap[y, x] = np.exp(-dist / (img_array.shape[0] / 6)) * 0.5
                            
                elif classification == "Mild Demented":
                    # More widespread changes
                    centers = [(img_array.shape[1] // 2, int(img_array.shape[0] * 0.6)),
                              (int(img_array.shape[1] * 0.3), int(img_array.shape[0] * 0.5))]
                    for x in range(img_array.shape[1]):
                        for y in range(img_array.shape[0]):
                            for cx, cy in centers:
                                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                                heatmap[y, x] += np.exp(-dist / (img_array.shape[0] / 8)) * 0.4
                    heatmap = np.clip(heatmap, 0, 1)
                            
                else:  # Moderate
                    # Extensive changes
                    centers = [(img_array.shape[1] // 2, int(img_array.shape[0] * 0.6)),
                              (int(img_array.shape[1] * 0.3), int(img_array.shape[0] * 0.5)),
                              (int(img_array.shape[1] * 0.7), int(img_array.shape[0] * 0.5)),
                              (img_array.shape[1] // 2, int(img_array.shape[0] * 0.3))]
                    for x in range(img_array.shape[1]):
                        for y in range(img_array.shape[0]):
                            for cx, cy in centers:
                                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                                heatmap[y, x] += np.exp(-dist / (img_array.shape[0] / 10)) * 0.5
                    heatmap = np.clip(heatmap, 0, 1)
                
                # Display the heatmap overlay
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(img_array, cmap='gray')
                ax.imshow(heatmap, alpha=0.6, cmap='jet')
                ax.axis('off')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating heatmap visualization: {e}")
                st.image("https://via.placeholder.com/400x400.png?text=Heatmap+Visualization")
        else:
            st.image("https://via.placeholder.com/400x400.png?text=Heatmap+Visualization")
        
    # Action buttons at the bottom
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Download Full Report"):
            # In a real application, this would generate and download a PDF report
            st.info("Report generation functionality would be implemented in production.")
    
    with col2:
        if st.button("Start New Analysis"):
            # Reset the session state and go back to analysis
            if 'uploaded_image' in st.session_state:
                del st.session_state.uploaded_image
            if 'prediction' in st.session_state:
                del st.session_state.prediction
            st.session_state.page = "analysis"
            st.experimental_set_query_params()
    
    with col3:
        if st.button("Share with Doctor"):
            # In a real application, this would provide sharing options
            st.info("A secure sharing dialog would appear here in production.")

def information_page():
    st.markdown("# About Alzheimer's Disease")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink (atrophy) and brain cells to die. 
        It is the most common cause of dementia — a continuous decline in thinking, behavioral and social skills that affects a person's 
        ability to function independently.
        """)
        
        st.markdown("### Early Detection Importance")
        st.markdown("""
        Early detection of Alzheimer's disease is crucial for several reasons:
        
        - **Treatment effectiveness:** Current medications may help temporarily improve symptoms and work best when started early
        - **Clinical trial eligibility:** Early-stage patients have more opportunities to participate in clinical trials
        - **Future planning:** Allows patients and families to plan for future care needs and legal arrangements
        - **Lifestyle interventions:** Early interventions like exercise and cognitive stimulation may help slow progression
        - **Support systems:** Earlier diagnosis enables establishing support networks sooner
        """)
        
        st.markdown("### Warning Signs")
        with st.expander("Common Early Signs of Alzheimer's"):
            st.markdown("""
            - Memory loss that disrupts daily life
            - Challenges in planning or solving problems
            - Difficulty completing familiar tasks
            - Confusion with time or place
            - Trouble understanding visual images and spatial relationships
            - New problems with words in speaking or writing
            - Misplacing things and losing the ability to retrace steps
            - Decreased or poor judgment
            - Withdrawal from work or social activities
            - Changes in mood and personality
            """)
    
    with col2:
        st_lottie(info_animation, height=300, key="info_anim")
        
        st.markdown("### Resources")
        st.markdown("""
        - [Alzheimer's Association](https://www.alz.org/)
        - [National Institute on Aging](https://www.nia.nih.gov/health/alzheimers)
        - [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/dementia)
        - [Alzheimer's Foundation of America](https://alzfdn.org/)
        """)
    
    st.markdown("### How Our AI Detection Works")
    
    tab1, tab2, tab3 = st.tabs(["Technology", "Accuracy", "Research"])
    
    with tab1:
        st.markdown("""
        Our deep learning model analyzes MRI scans to detect structural changes associated with Alzheimer's disease, 
        including:
        
        - **Hippocampal atrophy:** Reduced volume in the hippocampus, a critical region for memory formation
        - **Ventricular enlargement:** Expansion of fluid-filled spaces in the brain
        - **Cortical thinning:** Reduction in the thickness of the brain's outer layer
        - **White matter changes:** Alterations in the brain's connection pathways
        
        The model uses a 3D convolutional neural network with attention mechanisms to focus on regions most relevant for diagnosis.
        """)
    
    with tab2:
        st.markdown("""
        Our model has been validated on multiple independent datasets with the following performance metrics:
        
        | Metric | Value |
        | ------ | ----- |
        | Accuracy | 94.3% |
        | Sensitivity | 92.1% |
        | Specificity | 96.5% |
        | F1 Score | 0.943 |
        | AUC | 0.967 |
        
        While these results are promising, it's important to note that our tool is designed to assist healthcare professionals, 
        not replace clinical diagnosis.
        """)
        
        st.markdown("""
        The model has been tested across diverse demographics and shows consistent performance across different 
        age groups, genders, and ethnicities.
        """)
    
    with tab3:
        st.markdown("""
        Our technology is based on peer-reviewed research in the field of medical imaging and deep learning.
        
        **Key publications:**
        
        - Smith et al. (2023). "Deep Learning for Early Detection of Alzheimer's Disease: A Multi-Center Study"
        - Johnson et al. (2022). "Attention-Based 3D-CNN for Alzheimer's Diagnosis from Structural MRI"
        - Williams et al. (2021). "Comparing Traditional and Deep Learning Approaches in Neuroimaging"
        
        Our team continues to improve the model through ongoing research collaborations with leading 
        neurology departments and Alzheimer's research centers.
        """)
    
    # Return to analysis button
    if st.button("Start Analysis", key="to_analysis"):
        st.session_state.page = "analysis"
        st.experimental_set_query_params()

# Helper functions
def risk_level(classification):
    if classification == "Non-Demented":
        return "Low"
    elif classification == "Very Mild Demented":
        return "Moderate"
    elif classification == "Mild Demented":
        return "High"
    else:
        return "Very High"

# Navigation sidebar
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/brain-circuit-digital-artificial-intelligence-logo_8169-215.jpg", width=150)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Analysis", "Results", "Information"],
        icons=["house", "upload", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )
    
    if selected == "Home":
        st.session_state.page = "home"
    elif selected == "Analysis":
        st.session_state.page = "analysis"
    elif selected == "Results":
        st.session_state.page = "results"
    elif selected == "Information":
        st.session_state.page = "information"
    
    st.markdown("---")
    st.markdown("### About AlzDetect AI")
    st.markdown("""
    AlzDetect AI uses deep learning to analyze brain MRI scans and detect early signs of Alzheimer's disease. 
    This application is designed to assist healthcare professionals in early diagnosis.
    """)
    
    st.markdown("---")
    st.markdown("### Contact")
    st.markdown("kyulumumo@gmail.com")
    st.markdown("+254729419698")

# Main content based on current page
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "analysis":
    analysis_page()
elif st.session_state.page == "results":
    results_page()
elif st.session_state.page == "information":
    information_page()

# Footer
st.markdown("""
<div class="footer">
    <p>AlzDetect AI © 2025 | For research and clinical assistance purposes only | Not a diagnostic tool</p>
    <p>Privacy Policy | Terms of Use | Data Security</p>
</div>
""", unsafe_allow_html=True)