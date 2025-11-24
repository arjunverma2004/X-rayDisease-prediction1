echo '
# 🩺 X-ray Chest Classification App

An AI-powered application for classifying common thoracic pathologies from chest X-ray images. This project utilizes **Transfer Learning** with a fine-tuned Deep Learning model (specifically **EfficientNetB0**) and is deployed as a user-friendly web application using **Streamlit**.

The model is a multiclass classifier trained to detect findings typically associated with conditions like **Pneumonia**, **Tuberculosis**, and a **Normal** chest.

---

## ✨ Live Demo

Experience the app live and test it with your own chest X-ray images:

👉 **Live Streamlit App:** [https://x-raydisease-prediction.streamlit.app/](https://x-raydisease-prediction.streamlit.app/)

---

## 🚀 Model & Technology Stack

### Deep Learning Model
* **Architecture:** **EfficientNetB0** (a state-of-the-art Convolutional Neural Network).
* **Training Method:** **Transfer Learning**—The base EfficientNetB0 model was pre-trained on the massive **ImageNet** dataset, and its final layers were fine-tuned using a medical imaging dataset.
* **Dataset (Training Notebook):** The training process, as shown in `Chest_X_ray.ipynb`, uses a dataset containing images labeled for:
    * `normal`
    * `pneumonia`
    * `tuberculosis`

### Multi-Label Classification App (app.py)

The final deployed Streamlit application (`app.py`) demonstrates a **multi-label** classification approach, which is common in real-world chest X-ray analysis, potentially leveraging a model trained on a larger, multi-pathology dataset (like CheXpert or ChestX-ray14).

* **Model Name in App:** `grand_slam_best_model.keras`
* **Pathologies Predicted:** The app is configured to predict the likelihood of **14 distinct pathologies**:
    * `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Enlarged Cardiomediastinum`, `Fracture`, `Lung Lesion`, `Lung Opacity`, `No Finding`, `Pleural Effusion`, `Pleural Other`, `Pneumonia`, `Pneumothorax`, `Support Devices`

### Core Technologies
* **Frameworks:** **TensorFlow/Keras** for model development.
* **Web App:** **Streamlit** for rapid deployment and user interface.
* **Libraries:** `numpy`, `pandas`, `Pillow`.

---

## 📁 Repository Structure

| File/Folder | Description |
| :--- | :--- |
| `app.py` | The main Streamlit application code. Handles image upload, preprocessing, and prediction display. |
| `models/` | Directory containing the trained Keras model (`grand_slam_best_model.keras`). |
| `notebooks/` | Directory containing the training and analysis notebook (`Chest_X_ray.ipynb`). |
| `requirements.txt` | Lists all necessary Python dependencies (e.g., `streamlit`, `tensorflow`, `pandas`). |
| `Chest_X_ray.ipynb` | **Training Notebook:** Details the process of setting up the GPU, enabling mixed precision, loading the dataset, building the EfficientNetB0 model, and performing both frozen-base and fine-tuning steps. |

---

## 🛠️ Setup and Run Locally

### Prerequisites
* Python 3.8+
* Git

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd [YOUR_REPO_NAME]
    ```

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

The model was developed using a standard Transfer Learning workflow:

1.  **Data Preprocessing:** Images were resized to $(224, 224)$ pixels, and dataset pipelines (`tf.data.Dataset`) were configured for optimal performance using `AUTOTUNE` and `prefetching`.
2.  **Mixed Precision:** Training utilized **mixed_float16** to accelerate training on compatible GPUs (like the T4 in Colab/Kaggle).
3.  **Base Model Loading:** `EfficientNetB0` was loaded with pre-trained ImageNet weights, and its convolutional base was initially frozen (`base_model.trainable = False`).
4.  **Initial Training:** Only the new classification head (GlobalAveragePooling2D, Dropout, Dense) was trained for 10 epochs using the `adam` optimizer and `sparse_categorical_crossentropy` loss.
5.  **Fine-Tuning:** The base model was then unfrozen, and the entire model was recompiled with a **very low learning rate** (`1e-5`) to fine-tune the existing weights without corrupting the learned features. This phase used up to 20 epochs with `EarlyStopping` callbacks.
'
> README.md
