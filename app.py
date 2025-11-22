import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image

# IMPORTANT: Ensure TensorFlow behavior is consistent for prediction
# This line is added for robustness, although the code ran fine without it.
tf.compat.v1.enable_v2_tensorshape() 

# --- Configuration matching your notebook ---
IMAGE_SIZE = (320, 320) 
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

MODEL_PATH = 'models/grand_slam_best_model.keras'
THRESHOLD = 0.5 # Threshold for highlighting significant findings

# --- Functions ---

@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model using st.cache_resource for performance."""
    try:
        with st.spinner('Loading Deep Learning Model...'):
            # It's safer to use compile=False unless you know the custom objects needed
            model = load_model(model_path, compile=False) 
        st.success('Model loaded successfully! 🎉')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}. Ensure '{model_path}' is in the directory.")
        return None

def process_and_predict(uploaded_file, model):
    """Handles image processing and prediction."""
    try:
        # 1. Decode and Resize Image
        image = Image.open(uploaded_file).convert('RGB')

        # 2. Convert to NumPy Array and Normalize
        img_array = np.array(image.resize(IMAGE_SIZE), dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) # (1, 320, 320, 3)
        img_array = img_array / 255.0 

        # 3. Make Prediction
        with st.spinner('Analyzing X-ray image...'):
            logits = model.predict(img_array, verbose=0)
        
        # 4. Apply Sigmoid to get Probabilities
        probs = tf.nn.sigmoid(logits).numpy()[0] 
        
        # 5. Format Results
        results_df = pd.DataFrame({
            'Finding': LABELS,
            'Probability': probs
        }).sort_values(by='Probability', ascending=False)
        
        return image, results_df

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        return None, None

# --- Custom Styling for Results ---
def color_significant(val):
    """Adds background color for probabilities above the threshold."""
    # Use the global THRESHOLD variable
    global THRESHOLD 
    if isinstance(val, float):
        if val >= THRESHOLD:
            # High probability (e.g., Red/Orange tone)
            return 'background-color: rgba(255, 99, 71, 0.2)' 
        elif val >= 0.2:
            # Medium probability
            return 'background-color: rgba(255, 165, 0, 0.1)'
    return ''

# --- Main Streamlit App Logic ---

def main():
    st.set_page_config(
        page_title="DeepX-ray Classifier", 
        layout="wide", 
        initial_sidebar_state="expanded",
        menu_items=None
    )

    st.title("🩺 DeepX-ray Chest Pathology Classifier")
    st.markdown("A deep learning model (DenseNet121) for multi-label chest X-ray classification.")
    st.markdown("---")

    # Load Model (will only run once)
    model = load_keras_model(MODEL_PATH)
    if model is None:
        return

    # --- Sidebar for Upload ---
    st.sidebar.header("Step 1: Upload X-ray Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a chest X-ray image (JPG, JPEG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Note:** The model expects an anterior-posterior (AP) or posterior-anterior (PA) chest view."
    )

    # --- Main Content Area ---
    if uploaded_file is not None:
        
        image, results = process_and_predict(uploaded_file, model)
        
        if results is not None and image is not None:
            
            # --- Results Layout (Two Columns) ---
            col1, col2 = st.columns([1.5, 2.5]) 
            
            # Column 1: Image Display
            with col1:
                st.header("Uploaded Image")
                # Resize the displayed image using PIL before showing it to save on memory
                display_image = image.resize((400, int(image.height * 400 / image.width)))
                st.image(display_image, caption='Processed Image', use_column_width=True)
                
            # Column 2: Prediction Results
            with col2:
                st.header("Prediction Analysis")
                
                # 1. Critical Findings Alert
                critical_findings = results[results['Probability'] >= THRESHOLD]
                if not critical_findings.empty:
                    st.warning(
                        f"🚨 **Potential Critical Finding Detected!** ({len(critical_findings)} labels above {int(THRESHOLD*100)}% probability)."
                    )
                else:
                    st.success("👍 **No major findings detected** above the critical threshold.")
                
                st.subheader("All Pathologies")
                
                # Use st.dataframe with custom styling for better visual feedback
                st.dataframe(
                    results.style
                        .format({'Probability': "{:.2%}"})
                        .applymap(color_significant, subset=['Probability']),
                    hide_index=True,
                    use_container_width=True
                )
                
                # 2. Visualization
                st.subheader("Probability Visualization")
                # Filter down to the most likely results for cleaner chart
                chart_data = results[results['Probability'] >= 0.01] 
                st.bar_chart(
                    chart_data.set_index('Finding')['Probability'],
                    height=300
                )
                
    else:
        st.subheader("Awaiting Image Upload")
        st.markdown(
            """
            Please use the sidebar to upload a chest X-ray image (JPEG/PNG) 
            to start the automated classification process.
            """
        )
        


if __name__ == "__main__":
    main()