"""
Intelligent Plant Disease Detection System (IPDDS)
Model Training Pipeline for Potato & Tomato Diseases
Dataset: PlantVillage (Mohit Singh)
"""

import os
import json
import argparse
import hashlib
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


POTATO_TOMATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

METADATA_CLASS_NAMES = {
    "Tomato_Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_Early_blight": "Tomato___Early_blight",
    "Tomato_Late_blight": "Tomato___Late_blight",
    "Tomato_Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot": "Tomato___Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato_healthy": "Tomato___healthy"
}

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

def normalize_class_name(class_name):
    return "".join(character for character in class_name.casefold() if character.isalnum())


def dataset_source_name(path, fallback):
    normalized_path = path.replace("\\", "/").casefold()
    if "moazeldsokyx/plantvillage" in normalized_path:
        return "moazeldsokyx/plantvillage"
    if "emmarex/plantdisease" in normalized_path:
        return "emmarex/plantdisease"
    return fallback


CLASS_INDICES_BY_NORMALIZED_NAME = {
    normalize_class_name(class_name): index
    for index, class_name in enumerate(POTATO_TOMATO_CLASSES)
}


def has_target_classes(directory):
    if not os.path.isdir(directory):
        return False
    folder_names = {
        normalize_class_name(name)
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    }
    return all(name in folder_names for name in CLASS_INDICES_BY_NORMALIZED_NAME)


def resolve_source_directories(user_path):
    if not user_path or not os.path.isdir(user_path):
        raise FileNotFoundError(f"Dataset directory was not found: {user_path}")

    candidates = [
        user_path,
        os.path.join(user_path, "train"),
        os.path.join(user_path, "dataset"),
        os.path.join(user_path, "dataset", "train"),
        os.path.join(user_path, "PlantVillage"),
        os.path.join(user_path, "PlantVillage", "train")
    ]
    for child in os.listdir(user_path):
        child_path = os.path.join(user_path, child)
        if os.path.isdir(child_path):
            candidates.extend((
                child_path,
                os.path.join(child_path, "train"),
                os.path.join(child_path, "PlantVillage")
            ))

    for candidate in candidates:
        if has_target_classes(candidate):
            validation_directory = os.path.join(os.path.dirname(candidate), "val")
            return candidate, validation_directory if os.path.isdir(validation_directory) else None
    raise FileNotFoundError(
        f"Could not find all Potato and Tomato class folders beneath: {user_path}"
    )


def resolve_dataset_paths(user_path=None):
    """Locates the original PlantVillage dataset train and validation directories."""
    if user_path:
        return resolve_source_directories(user_path)

    cache_paths = [
        os.path.expanduser(r"~/.cache/kagglehub/datasets/moazeldsokyx/plantvillage/versions/1/dataset"),
        os.path.expanduser(r"~/.cache/kagglehub/datasets/mohitsingh1804/plantvillage/versions/1/PlantVillage")
    ]
    for cache_path in cache_paths:
        if os.path.isdir(cache_path):
            try:
                return resolve_source_directories(cache_path)
            except FileNotFoundError:
                continue
        
    try:
        from kagglehub import dataset_download
        downloaded = dataset_download("moazeldsokyx/plantvillage")
        return resolve_source_directories(downloaded)
    except Exception as error:
        raise RuntimeError("Could not locate or download the original PlantVillage dataset.") from error


def resolve_additional_dataset_paths(user_paths=None):
    """Locates the supplemental Kaggle PlantVillage dataset."""
    if user_paths is not None:
        return [resolve_source_directories(path) for path in user_paths]

    cache_path = os.path.expanduser(
        r"~/.cache/kagglehub/datasets/emmarex/plantdisease/versions/1"
    )
    if os.path.isdir(cache_path):
        try:
            return [resolve_source_directories(cache_path)]
        except FileNotFoundError:
            pass

    from kagglehub import dataset_download
    try:
        downloaded = dataset_download("emmarex/plantdisease")
        return [resolve_source_directories(downloaded)]
    except Exception as error:
        raise RuntimeError(
            "Could not locate or download the supplemental emmarex/plantdisease dataset."
        ) from error

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
    crop_metrics = crop_head.evaluate(val_features, val_crop_labels, verbose=0, return_dict=True)
    return crop_model, crop_metrics

def collect_examples(directory):
    examples = []
    class_counts = []
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    for label, class_name in enumerate(POTATO_TOMATO_CLASSES):
        source_class = os.path.join(directory, class_name)
        if not os.path.isdir(source_class):
            raise FileNotFoundError(f"Dataset class directory is missing: {source_class}")
        paths = sorted(
            os.path.join(source_class, file_name)
            for file_name in os.listdir(source_class)
            if file_name.lower().endswith(supported_extensions)
        )
        if not paths:
            raise ValueError(f"No supported images were found in: {source_class}")
        examples.extend((path, label) for path in paths)
        class_counts.append(len(paths))
    return examples, class_counts


def create_dataset(examples, batch_size, img_size, shuffle, seed):
    paths, labels = zip(*examples)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False
        )
        image.set_shape((None, None, 3))
        return tf.image.resize(image, img_size), label

    return dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def image_paths_for_class(directory, class_name):
    normalized_name = normalize_class_name(class_name)
    class_directory = next(
        (
            os.path.join(directory, folder_name)
            for folder_name in sorted(os.listdir(directory))
            if normalize_class_name(folder_name) == normalized_name
            and os.path.isdir(os.path.join(directory, folder_name))
        ),
        None
    )
    if class_directory is None:
        raise FileNotFoundError(f"Dataset class directory is missing: {class_name} in {directory}")
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    return sorted(
        os.path.join(class_directory, file_name)
        for file_name in os.listdir(class_directory)
        if file_name.lower().endswith(supported_extensions)
    )


def file_sha256(path, hash_cache):
    if path in hash_cache:
        return hash_cache[path]
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    image_hash = digest.digest()
    hash_cache[path] = image_hash
    return image_hash


def unique_image_paths(paths, seen_hashes, hash_cache):
    unique_paths = []
    duplicate_count = 0
    for path in paths:
        image_hash = file_sha256(path, hash_cache)
        if image_hash in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(image_hash)
        unique_paths.append(path)
    return unique_paths, duplicate_count


def create_stratified_splits(train_paths, val_paths, validation_split=0.2, seed=42):
    if isinstance(train_paths, str):
        train_paths = [train_paths]
    if val_paths is None:
        val_paths = []
    elif isinstance(val_paths, str):
        val_paths = [val_paths]
    train_paths = [path for path in train_paths if path]
    val_paths = [path for path in val_paths if path and os.path.isdir(path)]
    if not train_paths:
        raise ValueError("At least one training dataset directory is required.")

    rng = np.random.default_rng(seed)
    train_examples = []
    validation_examples = []
    train_counts = []
    validation_counts = []
    duplicate_count = 0
    hash_cache = {}

    for label, class_name in enumerate(POTATO_TOMATO_CLASSES):
        all_train_paths = [
            path
            for source_directory in train_paths
            for path in image_paths_for_class(source_directory, class_name)
        ]
        all_train_paths, source_duplicates = unique_image_paths(all_train_paths, set(), hash_cache)
        duplicate_count += source_duplicates
        if len(all_train_paths) < 2:
            raise ValueError(f"Class {class_name} needs at least two training images.")

        all_validation_paths = [
            path
            for source_directory in val_paths
            for path in image_paths_for_class(source_directory, class_name)
        ]
        if all_validation_paths:
            validation_paths, validation_duplicates = unique_image_paths(
                all_validation_paths,
                set(),
                hash_cache
            )
            duplicate_count += validation_duplicates
            validation_hashes = {
                file_sha256(path, hash_cache)
                for path in validation_paths
            }
            class_train_paths = []
            seen_training_hashes = set()
            for path in all_train_paths:
                image_hash = file_sha256(path, hash_cache)
                if image_hash in validation_hashes or image_hash in seen_training_hashes:
                    duplicate_count += 1
                    continue
                seen_training_hashes.add(image_hash)
                class_train_paths.append(path)
        else:
            rng.shuffle(all_train_paths)
            validation_count = max(1, int(round(len(all_train_paths) * validation_split)))
            validation_paths = all_train_paths[:validation_count]
            class_train_paths = all_train_paths[validation_count:]

        if not validation_paths or not class_train_paths:
            raise ValueError(f"Class {class_name} must contain training and validation images.")
        train_examples.extend((path, label) for path in class_train_paths)
        validation_examples.extend((path, label) for path in validation_paths)
        train_counts.append(len(class_train_paths))
        validation_counts.append(len(validation_paths))

    return train_examples, validation_examples, train_counts, validation_counts, duplicate_count


def export_tflite_models(disease_model, crop_model, assets_dir):
    os.makedirs(assets_dir, exist_ok=True)
    converted_models = {
        "disease_model.tflite": tf.lite.TFLiteConverter.from_keras_model(disease_model).convert(),
        "crop_type_model.tflite": tf.lite.TFLiteConverter.from_keras_model(crop_model).convert()
    }

    interpreters = {
        file_name: tf.lite.Interpreter(model_content=model_bytes)
        for file_name, model_bytes in converted_models.items()
    }
    for file_name, interpreter in interpreters.items():
        interpreter.allocate_tensors()
        input_shape = interpreter.get_input_details()[0]["shape"].tolist()
        if input_shape != [1, 128, 128, 3]:
            raise ValueError(f"{file_name} has an unexpected input shape: {input_shape}")
        expected_classes = 2 if file_name == "crop_type_model.tflite" else len(POTATO_TOMATO_CLASSES)
        output_shape = interpreter.get_output_details()[0]["shape"].tolist()
        if output_shape != [1, expected_classes]:
            raise ValueError(f"{file_name} has an unexpected output shape: {output_shape}")

    return converted_models


def train(
    dataset_dir=None,
    additional_dataset_dirs=None,
    epochs=10,
    batch_size=32,
    img_size=(128, 128),
    output_dir="model",
    android_assets_dir=os.path.join("android", "app", "src", "main", "assets")
):
    if img_size != (128, 128):
        raise ValueError("The Android and web apps share 128 by 128 models; keep --img-size at 128.")
    train_path, val_path = resolve_dataset_paths(dataset_dir)
    supplemental_sources = resolve_additional_dataset_paths(additional_dataset_dirs)
    train_paths = [train_path] + [source[0] for source in supplemental_sources]
    val_paths = ([val_path] if val_path else []) + [
        source[1] for source in supplemental_sources if source[1]
    ]

    tf.keras.utils.set_random_seed(42)
    train_examples, validation_examples, train_counts, validation_counts, duplicate_count = (
        create_stratified_splits(train_paths, val_paths or None)
    )
    train_ds = create_dataset(train_examples, batch_size, img_size, shuffle=True, seed=42)
    val_ds = create_dataset(validation_examples, batch_size, img_size, shuffle=False, seed=42)

    print("=" * 60)
    print("IPDDS - Potato & Tomato Leaf Disease Model Training")
    print("=" * 60)
    print(f"Training sources   : {train_paths}")
    print(f"Validation sources : {val_paths or 'stratified 80/20 split'}")
    print(f"Exact duplicates   : {duplicate_count}")
    print(f"Training images    : {len(train_examples)}")
    print(f"Validation images  : {len(validation_examples)}")
    print(f"Image Resolution   : {img_size}")
    print(f"Training by class  : {dict(zip(POTATO_TOMATO_CLASSES, train_counts))}")
    print(f"Validation by class: {dict(zip(POTATO_TOMATO_CLASSES, validation_counts))}")

    model = build_model(num_classes=len(POTATO_TOMATO_CLASSES), img_size=img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(output_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=".ipdds-training-", dir=output_dir)
    try:
        checkpoint_cb = callbacks.ModelCheckpoint(
            os.path.join(staging_dir, "best_disease_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        )
        early_stop_cb = callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[checkpoint_cb, early_stop_cb]
        )

        disease_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
        crop_model, crop_metrics = train_crop_classifier(model, train_ds, val_ds)
        training_datasets = [dataset_source_name(train_path, "primary PlantVillage dataset")]
        training_datasets.extend(
            dataset_source_name(source[0], "supplemental PlantVillage dataset")
            for source in supplemental_sources
        )
        metrics_export = {
            "validation_split": "stratified 80/20 per class after exact-file deduplication",
            "training_images": len(train_examples),
            "validation_images": len(validation_examples),
            "duplicate_images_removed": duplicate_count,
            "training_datasets": training_datasets,
            "disease_accuracy": float(disease_metrics["accuracy"]),
            "crop_accuracy": float(crop_metrics["accuracy"]),
            "crop_classes": ["Potato", "Tomato"],
            "class_validation_counts": dict(zip(POTATO_TOMATO_CLASSES, validation_counts))
        }
        metadata_export = {
            "class_names": POTATO_TOMATO_CLASSES,
            "classes": {
                class_name: DISEASE_METADATA[METADATA_CLASS_NAMES.get(class_name, class_name)]
                for class_name in POTATO_TOMATO_CLASSES
            },
            "image_size": list(img_size)
        }

        model.save(os.path.join(staging_dir, "disease_model.keras"))
        model.save(os.path.join(staging_dir, "disease_model.h5"))
        crop_model.save(os.path.join(staging_dir, "crop_type_model.keras"))
        with open(os.path.join(staging_dir, "class_indices.json"), "w", encoding="utf-8") as metadata_file:
            json.dump(metadata_export, metadata_file, indent=2)
        with open(os.path.join(staging_dir, "training_metrics.json"), "w", encoding="utf-8") as metrics_file:
            json.dump(metrics_export, metrics_file, indent=2)

        tflite_models = export_tflite_models(model, crop_model, android_assets_dir)
        staged_tflite = []
        for file_name, model_bytes in tflite_models.items():
            staged_path = os.path.join(android_assets_dir, f".{file_name}.tmp")
            with open(staged_path, "wb") as model_file:
                model_file.write(model_bytes)
            staged_tflite.append((staged_path, os.path.join(android_assets_dir, file_name)))
        with open(os.path.join(android_assets_dir, ".class_indices.json.tmp"), "w", encoding="utf-8") as metadata_file:
            json.dump(metadata_export, metadata_file, indent=2)
        staged_tflite.append((
            os.path.join(android_assets_dir, ".class_indices.json.tmp"),
            os.path.join(android_assets_dir, "class_indices.json")
        ))
        with open(os.path.join(android_assets_dir, ".training_metrics.json.tmp"), "w", encoding="utf-8") as metrics_file:
            json.dump(metrics_export, metrics_file, indent=2)
        staged_tflite.append((
            os.path.join(android_assets_dir, ".training_metrics.json.tmp"),
            os.path.join(android_assets_dir, "training_metrics.json")
        ))

        for file_name in (
            "disease_model.keras",
            "disease_model.h5",
            "crop_type_model.keras",
            "class_indices.json",
            "training_metrics.json"
        ):
            os.replace(
                os.path.join(staging_dir, file_name),
                os.path.join(output_dir, file_name)
            )
        for staged_path, target_path in staged_tflite:
            os.replace(staged_path, target_path)
    finally:
        for file_name in (
            ".disease_model.tflite.tmp",
            ".crop_type_model.tflite.tmp",
            ".class_indices.json.tmp",
            ".training_metrics.json.tmp"
        ):
            staged_path = os.path.join(android_assets_dir, file_name)
            if os.path.exists(staged_path):
                os.remove(staged_path)
        for file_name in os.listdir(staging_dir):
            staged_path = os.path.join(staging_dir, file_name)
            if os.path.isdir(staged_path):
                import shutil
                shutil.rmtree(staged_path)
            else:
                os.remove(staged_path)
        os.rmdir(staging_dir)

    print(f"[+] Held-out disease accuracy: {metrics_export['disease_accuracy']:.4f}")
    print(f"[+] Held-out crop accuracy: {metrics_export['crop_accuracy']:.4f}")
    print("[+] Keras, H5, metadata, training metrics, and Android TFLite models were updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Potato & Tomato Plant Disease Detector")
    parser.add_argument("--dataset-path", type=str, default=None, help="Custom path to PlantVillage dataset root or train folder")
    parser.add_argument(
        "--additional-dataset-path",
        action="append",
        default=None,
        help="Supplemental dataset root or train folder; repeat to add more sources"
    )
    parser.add_argument(
        "--skip-supplemental-dataset",
        action="store_true",
        help="Train only on the primary PlantVillage dataset"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=128, help="Square image resolution (e.g. 128 for fast CPU training)")
    args = parser.parse_args()
    
    train(
        dataset_dir=args.dataset_path,
        additional_dataset_dirs=[] if args.skip_supplemental_dataset else args.additional_dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size)
    )
