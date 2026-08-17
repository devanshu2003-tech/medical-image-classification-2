import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Set Streamlit page config
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .prediction-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #334155;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: 800;
        color: #0F766E;
        margin: 0.5rem 0;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #CCFBF1;
        color: #0F766E;
        font-weight: 700;
        padding: 0.4rem 0.8rem;
        border-radius: 9999px;
        font-size: 1rem;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 2rem;
        color: #991B1B;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Class names mapping
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
CLASS_KEYS = ['glioma', 'meningioma', 'notumor', 'pituitary']

MODEL_PATHS = [
    os.path.join("models", "final_brain_tumor_model.keras"),
    os.path.join("models", "brain_tumor_model.keras")
]

@st.cache_resource
def load_keras_model():
    """Load the Keras model with fallback path checking."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
    return None, None

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Preprocess PIL image for CNN prediction."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    resized_img = image.resize(target_size)
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=70)
    st.title("Brain MRI Classifier")
    st.markdown("---")
    st.subheader("ℹ️ Project Info")
    st.write("""
    This deep learning application classifies Brain MRI scans into four distinct categories:
    - **Glioma Tumor**
    - **Meningioma Tumor**
    - **No Tumor**
    - **Pituitary Tumor**
    """)
    
    st.markdown("---")
    st.subheader("⚙️ Model Info")
    model, loaded_path = load_keras_model()
    if model:
        st.success(f"Model Loaded Successfully!")
        st.caption(f"📁 Source: `{os.path.basename(loaded_path)}`")
    else:
        st.error("Model file not found in `models/` directory.")

    st.markdown("---")
    st.caption("Developed with TensorFlow & Streamlit for Medical Image Classification")

# --- Main App Header ---
st.markdown('<div class="main-header">🧠 Brain Tumor MRI Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a Brain MRI image to get real-time AI-powered classification</div>', unsafe_allow_html=True)

# Main UI layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload MRI Scan")
    
    input_source = st.radio("Select Input Method:", ["Upload Image File", "Use Sample Test Image"], horizontal=True)
    
    uploaded_file = None
    if input_source == "Upload Image File":
        uploaded_file = st.file_uploader("Choose an MRI image (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    else:
        sample_path = os.path.join("sample_images", "Test.jpg")
        if os.path.exists(sample_path):
            uploaded_file = sample_path
            st.info("Loaded sample image `sample_images/Test.jpg`")
        else:
            st.warning("Sample image not found in `sample_images/Test.jpg`")

    image_to_process = None
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, str):
                image_to_process = Image.open(uploaded_file)
            else:
                image_to_process = Image.open(uploaded_file)
            
            st.image(image_to_process, caption="Uploaded MRI Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

with col2:
    st.subheader("📊 Diagnostic Output")
    
    if image_to_process is not None:
        if model is None:
            st.error("Cannot perform prediction because the model is not loaded.")
        else:
            with st.spinner("Analyzing MRI scan..."):
                # Preprocess image
                processed_img = preprocess_image(image_to_process)
                
                # Make prediction
                predictions = model.predict(processed_img)[0]
                pred_idx = np.argmax(predictions)
                predicted_label = CLASS_NAMES[pred_idx]
                confidence = float(predictions[pred_idx] * 100)

            # Highlight card
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-title">Predicted Classification</div>
                    <div class="prediction-result">{predicted_label}</div>
                    <span class="confidence-badge">Confidence: {confidence:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Probability Breakdown")
            
            # Display bar charts for all classes
            for idx, (label, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                prob_pct = float(prob * 100)
                st.write(f"**{label}**: `{prob_pct:.2f}%`")
                st.progress(min(float(prob), 1.0))
    else:
        st.info("👆 Please upload an MRI image or select the sample image on the left to see prediction results.")

# --- Disclaimer ---
st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Disclaimer:</strong> This application is intended solely for educational and research purposes.
        It is not designed or intended to be used as a clinical diagnostic tool or to replace professional medical advice.
    </div>
""", unsafe_allow_html=True)
