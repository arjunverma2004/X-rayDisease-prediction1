import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import pandas as pd
# --- Configuration matching your notebook ---
IMAGE_SIZE = (320, 320) 
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# --- Functions ---

@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model using st.cache_resource for performance."""
    # Note: Use 'allow_partial_model_loading=False' if needed, but the default should work for your saved .keras file
    with st.spinner('Model is loading...'):
        model = load_model(model_path, compile=False) # Compile=False if using a custom loss/metric
    return model

def process_and_predict(uploaded_file, model):
    """Handles image processing and prediction."""
    try:
        # 1. Decode and Resize Image
        # Streamlit's file_uploader returns a BytesIO stream
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize(IMAGE_SIZE)
        st.image(image, caption='Uploaded X-ray Image', use_column_width=True)

        # 2. Convert to NumPy Array and Normalize (Matching notebook's process_img)
        img_array = np.array(image, dtype=np.float32)
        # Convert to a batch format (1, 320, 320, 3)
        img_array = np.expand_dims(img_array, axis=0) 
        # Normalize
        img_array = img_array / 255.0 

        # 3. Make Prediction
        st.write("### Making Prediction...")
        logits = model.predict(img_array)
        
        # 4. Apply Sigmoid to get Probabilities (Matching notebook's inference)
        probs = tf.nn.sigmoid(logits).numpy()[0] 
        
        # 5. Format Results
        results_df = pd.DataFrame({
            'Finding': LABELS,
            'Probability': probs
        }).sort_values(by='Probability', ascending=False)
        
        return results_df

    except Exception as e:
        st.error(f"Error during processing or prediction: {e}")
        return None

# --- Main Streamlit App Logic ---

def main():
    st.set_page_config(page_title="X-ray Chest Classification App 🩺", layout="wide")
    st.title("X-ray Chest Classification App")
    st.markdown("Upload a chest X-ray image to get predictions for common pathologies.")
    
    # Load Model (will only run once thanks to @st.cache_resource)
    model = load_keras_model('grand_slam_best_model.keras')

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a chest X-ray image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Perform prediction and display results
        results = process_and_predict(uploaded_file, model)
        
        if results is not None:
            st.header("Prediction Results")
            # Show the top 5 most likely findings
            top_n = 5
            st.subheader(f"Top {top_n} Predicted Findings")
            st.dataframe(results.head(top_n).style.format({'Probability': "{:.2%}"}))
            
            # Optionally show a bar chart of probabilities
            st.bar_chart(results.set_index('Finding')['Probability'])

    else:
        st.info("Awaiting file upload. Please upload an image to begin classification.")
        
if __name__ == "__main__":
    main()