# 🩺 X-ray Chest Classification App

An **AI-powered application** for classifying common thoracic pathologies from chest X-ray images.  
This project utilizes **Transfer Learning** with a fine-tuned Deep Learning model (**DenseNet121**) and is deployed as a user-friendly web application using **Streamlit**.

The model is a **multi-label classifier** trained to detect multiple findings commonly associated with chest-related conditions, offering a comprehensive assessment.

---

## ✨ Live Demo

Experience the app live and test it with your own chest X-ray images:

👉 **Live Streamlit App:** [https://x-raydisease-prediction.streamlit.app/](https://x-raydisease-prediction.streamlit.app/)

---

## 👨‍💻 Contributors

| Role | Name | GitHub | LinkedIn |
|------|------|---------|-----------|
| **Data Science & Model Training** | **Arjun Verma** | @arjunverma2004 | https://www.linkedin.com/in/arjunverma2004 |
| **Streamlit Deployment** | **Aaditya Singh** | @AadityaSinghWeb2005 | https://www.linkedin.com/in/aaditya-singh-bbab812a9 |

---

## 🚀 Model & Technology Stack

### **Deep Learning Model (Trained by Arjun Verma)**

- Training process is detailed in the **grand-xray-2.ipynb** notebook.  
- **Architecture:** DenseNet121 (Pre-trained on ImageNet)  
- **Classification Type:** Multi-Label (14 distinct pathologies)  
- **Model Used in App:** `grand_slam_best_model.keras`

### **Pathologies Predicted**

Atelectasis, Cardiomegaly, Consolidation, Edema,  
Enlarged Cardiomediastinum, Fracture, Lung Lesion,  
Lung Opacity, No Finding, Pleural Effusion, Pleural Other,  
Pneumonia, Pneumothorax, Support Devices

---

### **Core Technologies**

- **Frameworks:** TensorFlow/Keras  
- **Web App:** Streamlit  
- **Libraries:** numpy, pandas, Pillow  

---

## 📁 Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `app.py` | Main Streamlit app (by Aaditya Singh). Handles image upload, preprocessing, predictions. |
| `models/` | Directory containing the trained model (`grand_slam_best_model.keras`). |
| `grand-xray-2.ipynb` | Training notebook detailing dataset setup, DenseNet121 model training, and prediction generation. |
| `requirements.txt` | Python dependencies (tensorflow, streamlit, pandas, etc.). |

---

## 🛠️ Setup and Run Locally

### **Prerequisites**
- Python 3.8+
- Git

### **Steps**

**1. Clone the repository:**
```bash
git clone [YOUR_REPO_URL]
cd [YOUR_REPO_NAME]


2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The requirements file should include `tensorflow` and `streamlit`.)*

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

The application will automatically open in your web browser, typically at `http://localhost:8501`.

---

## 🎓 Training Process Highlights

### **1. Data Preparation**
- Resize: **320×320**  
- Normalization  
- Efficient pipelines via `tf.data.Dataset`, AUTOTUNE, prefetch  

### **2. Model Building**
- Backbone: **DenseNet121**  
- Head: GAP → Dropout → Dense  

### **3. Optimization**
- Optimizer: **Adam**  
- Loss: **Binary Crossentropy**  
- Metric: **AUC**

### **4. Callbacks**
- **ModelCheckpoint**  
- **ReduceLROnPlateau**

---
