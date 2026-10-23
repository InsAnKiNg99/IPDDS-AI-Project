# IPDDS: Intelligent Plant Disease Detection System 🌿

IPDDS is a deep learning web application built to identify plant leaf diseases in real-time. This model is trained on the **PlantVillage** dataset, targeting **13 distinct disease and health conditions** in **Tomato** and **Potato** crops.

---

## 🎯 Target Categories (13 Classes)

### 🥔 Potato Classes
- `Potato - Early Blight` (*Alternaria solani*)
- `Potato - Late Blight` (*Phytophthora infestans*)
- `Potato - Healthy`

### 🍅 Tomato Classes
- `Tomato - Bacterial Spot` (*Xanthomonas spp.*)
- `Tomato - Early Blight` (*Alternaria linariae*)
- `Tomato - Late Blight` (*Phytophthora infestans*)
- `Tomato - Leaf Mold` (*Passalora fulva*)
- `Tomato - Septoria Leaf Spot` (*Septoria lycopersici*)
- `Tomato - Spider Mites (Two-Spotted)` (*Tetranychus urticae*)
- `Tomato - Target Spot` (*Corynespora cassiicola*)
- `Tomato - Tomato Yellow Leaf Curl Virus` (*TYLCV*)
- `Tomato - Tomato Mosaic Virus` (*ToMV*)
- `Tomato - Healthy`

---

## 🚀 Quickstart Guide

### 1. Requirements & Setup
Ensure dependencies are installed:
```bash
pip install tensorflow streamlit pillow numpy kagglehub
```

### 2. Dataset Setup
The dataset is downloaded via `kagglehub`:
```bash
python dataset/down.py
```
This downloads `mohitsingh1804/plantvillage` directly into your local cache (`~/.cache/kagglehub/...`).

### 3. Model Training
To train the CNN model specifically on Potato and Tomato:
```bash
python train.py --epochs 10 --batch-size 32
```
You can also run through the interactive Jupyter notebook:
- Open [`notebook/model_training_v2.ipynb`](notebook/model_training_v2.ipynb) to train the disease and crop-type classifiers on the Tomato and Potato dataset.
- The v2 notebook makes a separate 80/20 split inside every disease-class folder so both Potato and Tomato images are present during training and validation.

Trained models are saved to:
- `model/disease_model.h5`
- `model/disease_model.keras`
- `model/crop_type_model.keras`
- `model/class_indices.json`

### 4. Running the Web Application
Launch the Streamlit app:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, upload an image of a Potato or Tomato leaf, and inspect the detected crop, disease diagnosis, confidence breakdown, and care recommendations. If auto-detection is uncertain for an unusual photo, use the crop-identification selector to choose Potato or Tomato; the disease prediction and treatment are then restricted to that crop.
