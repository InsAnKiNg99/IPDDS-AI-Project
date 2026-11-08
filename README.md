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

## 🚀 IPDDS 1.1

Version 1.1 adds the Potato and Tomato images from Kaggle's [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) to the existing training data. The app still predicts the same 13 Potato and Tomato classes; other crop classes are not included. Exact duplicate image files are removed before making the stratified train/validation split.

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
This downloads `moazeldsokyx/plantvillage` directly into your local cache (`~/.cache/kagglehub/...`).

### 3. Model Training
To train the MobileNetV2 disease model on Potato and Tomato. The default input size is 160 by 160:
```bash
python train.py --dataset-path "C:\Users\Muneer Hameed\.cache\kagglehub\datasets\moazeldsokyx\plantvillage\versions\1\dataset\train" --additional-dataset-path "C:\Users\Muneer Hameed\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1" --epochs 10 --batch-size 32
```
The training command downloads the supplemental dataset from KaggleHub when it is not already cached. It combines matching Potato and Tomato classes, removes byte-identical image files, and creates a deterministic, class-stratified 80/20 holdout when no separate validation directory exists. It writes the shared Keras models, class metadata, holdout metrics, and matching TensorFlow Lite models into the Android assets directory so the web and Android apps use the same trained weights. Pass `--skip-supplemental-dataset` to train using only the original dataset.
You can also run through the interactive Jupyter notebook:
- Open [`notebook/model_training_v2.ipynb`](notebook/model_training_v2.ipynb) to train the disease and crop-type classifiers on the Tomato and Potato dataset.
- The v2 notebook makes a separate 80/20 split inside every disease-class folder so both Potato and Tomato images are present during training and validation.

Trained models are saved to:
- `model/disease_model.h5`
- `model/disease_model.keras`
- `model/crop_type_model.keras`
- `model/class_indices.json`
- `model/training_metrics.json`

### 4. Running the Web Application
Launch the Streamlit app:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, upload a leaf image, and inspect the estimated crop, disease diagnosis, confidence breakdown, and care recommendations. Crop type is estimated automatically; when the model is unsure, the app asks for a clearer photo instead of letting the user override the estimate.

### 5. Native Android App
Open the [`android`](android) folder in Android Studio and build the debug APK. The app bundles both trained TensorFlow Lite models and matching disease/treatment metadata, so image checks work offline. It supports Android 7.0 and newer.

See [`android/README.md`](android/README.md) for build, install, and image-preprocessing details.
