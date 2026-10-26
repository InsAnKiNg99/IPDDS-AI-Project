"""
Intelligent Plant Disease Detection System (IPDDS)
Model Training Pipeline for Potato & Tomato Diseases
Dataset: PlantVillage (Mohit Singh)
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


POTATO_TOMATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

DISEASE_METADATA = {
    "Potato___Early_blight": {
        "display_name": "Potato: Early Blight",
        "crop": "Potato",
        "status": "Diseased",
        "pathogen": "Fungus (Alternaria solani)",
        "recommendation": "Remove lower infected leaves, ensure adequate crop rotation, and apply copper-based fungicides."
    },
    "Potato___Late_blight": {
        "display_name": "Potato: Late Blight",
        "crop": "Potato",
        "status": "Diseased",
        "pathogen": "Water mold (Phytophthora infestans)",
        "recommendation": "High severity. Destroy infected plants immediately, improve soil drainage, and avoid wet foliage."
    },
    "Potato___healthy": {
        "display_name": "Potato: Healthy",
        "crop": "Potato",
        "status": "Healthy",
        "pathogen": "None",
        "recommendation": "Leaves appear completely healthy. Continue standard watering and fertilizing schedules."
    },
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato: Bacterial Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Bacteria (Xanthomonas spp.)",
        "recommendation": "Avoid overhead watering, sanitize tools between pruning, and apply fixed copper sprays."
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato: Early Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Fungus (Alternaria linariae)",
        "recommendation": "Mulch the soil surface, prune bottom leaves to stop splash-back, and use organic neem or fungicides."
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato: Late Blight",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "recommendation": "Act quickly: prune damaged vines, keep canopy aerated, and apply preventative bio-fungicides."
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato: Leaf Mold",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Fungus (Passalora fulva)",
        "recommendation": "Improve ventilation, lower relative humidity below 85%, and use resistant tomato varieties."
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato: Septoria Leaf Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Fungus (Septoria lycopersici)",
        "recommendation": "Remove infected lower foliage, avoid wetting foliage during irrigation, and rotate crops."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato: Spider Mites",
        "crop": "Tomato",
        "status": "Infested",
        "pathogen": "Pest (Tetranychus urticae)",
        "recommendation": "Apply insecticidal soap or horticultural oils under leaves. Release predatory mites if indoors."
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato: Target Spot",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Fungus (Corynespora cassiicola)",
        "recommendation": "Improve airflow between plants, keep foliage dry, and apply targeted copper/mancozeb fungicides."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato: Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Virus (TYLCV transmitted by Whiteflies)",
        "recommendation": "Use yellow sticky traps and insect netting against whiteflies. Remove and burn heavily infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato: Mosaic Virus",
        "crop": "Tomato",
        "status": "Diseased",
        "pathogen": "Tobamovirus (Mechanically transmitted)",
        "recommendation": "Disinfect tools with 10% bleach, avoid handling plants when damp, and remove infected plants."
    },
    "Tomato___healthy": {
        "display_name": "Tomato: Healthy",
        "crop": "Tomato",
        "status": "Healthy",
        "pathogen": "None",
        "recommendation": "Leaves appear vibrant and healthy. Maintain consistent moisture and balanced nitrogen levels."
    }
}

def resolve_dataset_paths(user_path=None):
    """Locates train and val directories for PlantVillage."""
    if user_path and os.path.exists(user_path):
        train_dir = os.path.join(user_path, "train") if os.path.exists(os.path.join(user_path, "train")) else user_path
        val_dir = os.path.join(user_path, "val") if os.path.exists(os.path.join(user_path, "val")) else None
        return train_dir, val_dir


    cache_path = os.path.expanduser(r"~/.cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1/PlantVillage")
    if os.path.exists(cache_path):
        return os.path.join(cache_path, "train"), os.path.join(cache_path, "val")
        
    try:
        import kagglehub
        downloaded = kagglehub.dataset_download("mohitsingh1804/plantvillage")
        pv_sub = os.path.join(downloaded, "PlantVillage")
        root = pv_sub if os.path.exists(pv_sub) else downloaded
        return os.path.join(root, "train"), os.path.join(root, "val")
    except Exception as e:
        print(f"[!] kagglehub automatic lookup failed: {e}")
        return None, None

def build_model(num_classes, img_size=(128, 128)):
    """
    Builds a robust, lightweight Convolutional Neural Network with:
    - Data Augmentation layers to prevent overfitting
    - Conv2D + BatchNormalization + MaxPooling2D blocks
    - Dropout + Dense classification head
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    

    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1./255)(x)
    

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="IPDDS_Potato_Tomato_CNN")
    return model

def train_crop_classifier(model, train_ds, val_ds):
    feature_extractor = models.Model(
        inputs=model.input,
        outputs=model.layers[-3].output,
        name="IPDDS_Feature_Extractor"
    )
    crop_labels_by_class = tf.constant(
        [0 if class_name.startswith("Potato") else 1 for class_name in POTATO_TOMATO_CLASSES],
        dtype=tf.int32
    )

    def extract_features(dataset):
        feature_batches = []
        crop_label_batches = []
        for images, disease_labels in dataset:
            feature_batches.append(feature_extractor(images, training=False).numpy())
            crop_label_batches.append(tf.gather(crop_labels_by_class, disease_labels).numpy())
        return np.concatenate(feature_batches), np.concatenate(crop_label_batches)

    train_features, train_crop_labels = extract_features(train_ds)
    val_features, val_crop_labels = extract_features(val_ds)
    crop_counts = np.bincount(train_crop_labels, minlength=2)
    crop_class_weights = {
        index: len(train_crop_labels) / (2 * count)
        for index, count in enumerate(crop_counts)
    }

    crop_head = models.Sequential([
        layers.Input(shape=(feature_extractor.output_shape[-1],)),
        layers.Dense(2, activation="softmax")
    ], name="IPDDS_Crop_Head")
    crop_head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    crop_head.fit(
        train_features,
        train_crop_labels,
        validation_data=(val_features, val_crop_labels),
        class_weight=crop_class_weights,
        epochs=30,
        batch_size=128,
        callbacks=[callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True
        )]
    )

    crop_model = models.Model(
        inputs=model.input,
        outputs=crop_head(feature_extractor.output),
        name="IPDDS_Crop_Classifier"
    )
    crop_model.save(os.path.join("model", "crop_type_model.keras"))
    print("[+] Crop classifier saved to model/crop_type_model.keras (index 0: Potato, index 1: Tomato)")

def train(dataset_dir=None, epochs=10, batch_size=32, img_size=(128, 128)):
    train_path, val_path = resolve_dataset_paths(dataset_dir)
    print("=" * 60)
    print("IPDDS - Potato & Tomato Leaf Disease Model Training")
    print("=" * 60)
    print(f"Train Dataset Path : {train_path}")
    print(f"Val Dataset Path   : {val_path}")
    print(f"Image Resolution   : {img_size}")
    
    if not train_path or not os.path.exists(train_path):
        print(f"[Error] Training directory not found: {train_path}")
        sys.exit(1)

    print(f"\nLoading {len(POTATO_TOMATO_CLASSES)} Potato & Tomato classes...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        class_names=POTATO_TOMATO_CLASSES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    if val_path and os.path.exists(val_path):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_path,
            labels="inferred",
            class_names=POTATO_TOMATO_CLASSES,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
    else:

        val_size = int(0.2 * len(train_ds))
        val_ds = train_ds.take(val_size)
        train_ds = train_ds.skip(val_size)


    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    if val_ds:
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


    model = build_model(num_classes=len(POTATO_TOMATO_CLASSES), img_size=img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()


    os.makedirs("model", exist_ok=True)
    h5_path = os.path.join("model", "disease_model.h5")
    keras_path = os.path.join("model", "disease_model.keras")
    
    checkpoint_cb = callbacks.ModelCheckpoint(
        h5_path,
        save_best_only=True,
        monitor="val_accuracy" if val_ds else "accuracy",
        mode="max"
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy" if val_ds else "accuracy",
        patience=4,
        restore_best_weights=True
    )
    

    metadata_export = {
        "class_names": POTATO_TOMATO_CLASSES,
        "classes": DISEASE_METADATA,
        "image_size": list(img_size)
    }
    with open(os.path.join("model", "class_indices.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_export, f, indent=2)
    print("\n[+] Saved class indices and metadata to model/class_indices.json")

    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb]
    )


    model.save(keras_path)
    model.save(h5_path)
    print(f"\n[+] Training complete! Model saved to:\n  - {h5_path}\n  - {keras_path}")
    train_crop_classifier(model, train_ds, val_ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Potato & Tomato Plant Disease Detector")
    parser.add_argument("--dataset-path", type=str, default=None, help="Custom path to PlantVillage dataset root or train folder")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=128, help="Square image resolution (e.g. 128 for fast CPU training)")
    args = parser.parse_args()
    
    train(
        dataset_dir=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size)
    )
