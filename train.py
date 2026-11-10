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
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
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
    if "mohitsingh1804/plantvillage" in normalized_path:
        return "mohitsingh1804/plantvillage"
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
        print(f"[!] Supplemental dataset skipped: {error}")
        return []

def letterbox(image, img_size):
    image = tf.cast(image, tf.float32)
    return tf.image.resize_with_pad(image, img_size[0], img_size[1])


def augment_training_image(image, img_size):
    """Letterbox, then vary leaf scale so small leaves and close-ups are learned."""
    side = img_size[0]
    image = letterbox(image, img_size)
    choice = tf.random.uniform([])

    def shrink():
        scale = tf.random.uniform([], 0.35, 0.80)
        new_side = tf.cast(tf.round(scale * side), tf.int32)
        small = tf.image.resize(image, [new_side, new_side])
        offset = tf.random.uniform([], 0, side - new_side + 1, dtype=tf.int32)
        return tf.image.pad_to_bounding_box(small, offset, offset, side, side)

    def zoom():
        scale = tf.random.uniform([], 1.25, 2.00)
        crop_side = tf.cast(tf.round(tf.cast(side, tf.float32) / scale), tf.int32)
        crop_side = tf.clip_by_value(crop_side, 32, side)
        max_offset = side - crop_side
        offset = tf.random.uniform([], 0, max_offset + 1, dtype=tf.int32)
        cropped = tf.image.crop_to_bounding_box(image, offset, offset, crop_side, crop_side)
        return tf.image.resize(cropped, [side, side])

    image = tf.cond(
        choice < 0.55,
        lambda: image,
        lambda: tf.cond(choice < 0.80, shrink, zoom),
    )
    image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 28.0)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    return tf.clip_by_value(image, 0.0, 255.0)


def build_model(num_classes, img_size=(160, 160)):
    """
    ImageNet MobileNetV2 backbone with a small disease head.
    Pixels stay in 0..255 until the rescaling layer, matching both apps.
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        alpha=1.0,
        name="mobilenet_v2_backbone",
    )
    backbone.trainable = False
    embedding = layers.Activation("linear", name="embedding")(backbone(x))
    hidden = layers.Dense(256, activation="relu", name="disease_hidden")(embedding)
    hidden = layers.Dropout(0.30, name="disease_dropout")(hidden)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease")(hidden)
    model = models.Model(inputs=inputs, outputs=outputs, name="IPDDS_MobileNetV2")
    return model


def compile_disease_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_backbone_top(model, trainable_layers):
    backbone = model.get_layer("mobilenet_v2_backbone")
    backbone.trainable = True
    frozen_count = max(0, len(backbone.layers) - trainable_layers)
    for layer in backbone.layers[:frozen_count]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return backbone


def disease_class_weights(train_counts, max_weight=4.0):
    counts = np.asarray(train_counts, dtype=np.float64)
    raw_weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    capped = np.minimum(raw_weights, max_weight)
    return {index: float(weight) for index, weight in enumerate(capped)}


def classification_scores(labels, predictions, class_names):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    report = {}
    f1_scores = []
    recalls = []
    for index, class_name in enumerate(class_names):
        true_positive = int(np.sum((labels == index) & (predictions == index)))
        support = int(np.sum(labels == index))
        predicted_count = int(np.sum(predictions == index))
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
            "support": support,
        }
        f1_scores.append(f1_score)
        recalls.append(recall)
    return report, float(np.mean(recalls)), float(np.mean(f1_scores))

def train_crop_classifier(model, train_ds, val_ds):
    feature_extractor = models.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output,
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


def create_dataset(examples, batch_size, img_size, shuffle, seed, training):
    paths, labels = zip(*examples)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    options = tf.data.Options()
    options.deterministic = True
    dataset = dataset.with_options(options)
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False
        )
        image.set_shape((None, None, 3))
        image = tf.cast(image, tf.float32)
        if training:
            image = augment_training_image(image, img_size)
        else:
            image = letterbox(image, img_size)
        return image, label

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


def export_tflite_models(disease_model, crop_model, assets_dir, img_size):
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
        if input_shape != [1, img_size[0], img_size[1], 3]:
            raise ValueError(f"{file_name} has an unexpected input shape: {input_shape}")
        expected_classes = 2 if file_name == "crop_type_model.tflite" else len(POTATO_TOMATO_CLASSES)
        output_shape = interpreter.get_output_details()[0]["shape"].tolist()
        if output_shape != [1, expected_classes]:
            raise ValueError(f"{file_name} has an unexpected output shape: {output_shape}")

    return converted_models


def train(
    dataset_dir=None,
    additional_dataset_dirs=None,
    epochs=8,
    batch_size=32,
    img_size=(160, 160),
    output_dir="model",
    android_assets_dir=os.path.join("android", "app", "src", "main", "assets")
):
    if img_size[0] != img_size[1] or img_size[0] < 128:
        raise ValueError("Use a square image size of at least 128. The trained default is 160.")
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
    train_ds = create_dataset(
        train_examples, batch_size, img_size, shuffle=True, seed=42, training=True
    )
    train_clean_ds = create_dataset(
        train_examples, batch_size, img_size, shuffle=False, seed=42, training=False
    )
    val_ds = create_dataset(
        validation_examples, batch_size, img_size, shuffle=False, seed=42, training=False
    )

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

    class_weights = disease_class_weights(train_counts)
    print(f"Class weights      : {class_weights}")
    model = build_model(num_classes=len(POTATO_TOMATO_CLASSES), img_size=img_size)
    compile_disease_model(model, learning_rate=0.001)

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
            patience=3,
            restore_best_weights=True
        )

        class KeepBestWeights(callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best = -1.0
                self.weights = None

            def on_epoch_end(self, epoch, logs=None):
                score = float((logs or {}).get("val_accuracy", -1.0))
                if score > self.best:
                    self.best = score
                    self.weights = self.model.get_weights()

        keep_best = KeepBestWeights()

        print("Phase 1: training the disease head on a frozen MobileNetV2 backbone")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=[keep_best, checkpoint_cb, early_stop_cb, callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
                mode="max",
                verbose=1,
            )]
        )

        print("Phase 2: fine-tuning the top of MobileNetV2")
        unfreeze_backbone_top(model, trainable_layers=40)
        compile_disease_model(model, learning_rate=5e-5)
        finetune_epochs = max(3, epochs // 2)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=finetune_epochs,
            class_weight=class_weights,
            callbacks=[keep_best, checkpoint_cb, callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=2,
                restore_best_weights=True,
            ), callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
                mode="max",
                verbose=1,
            )]
        )
        if keep_best.weights is not None:
            model.set_weights(keep_best.weights)
            print(f"Restored best validation accuracy: {keep_best.best:.4f}")

        disease_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
        disease_probabilities = model.predict(val_ds, verbose=0)
        disease_predictions = np.argmax(disease_probabilities, axis=1)
        validation_labels = np.asarray([label for _, label in validation_examples], dtype=np.int32)
        if len(disease_predictions) != len(validation_labels):
            raise ValueError("Validation predictions do not match the validation labels.")
        class_report, macro_recall, macro_f1 = classification_scores(
            validation_labels,
            disease_predictions,
            POTATO_TOMATO_CLASSES,
        )
        crop_model, crop_metrics = train_crop_classifier(model, train_clean_ds, val_ds)
        training_datasets = [dataset_source_name(train_path, "primary PlantVillage dataset")]
        training_datasets.extend(
            dataset_source_name(source[0], "supplemental PlantVillage dataset")
            for source in supplemental_sources
        )
        metrics_export = {
            "validation_split": (
                "official validation directories after exact-file deduplication"
                if val_paths
                else "stratified 80/20 per class after exact-file deduplication"
            ),
            "training_images": len(train_examples),
            "validation_images": len(validation_examples),
            "duplicate_images_removed": duplicate_count,
            "training_datasets": training_datasets,
            "architecture": "MobileNetV2 ImageNet backbone, 256-unit disease head, top-layer fine-tune",
            "image_size": list(img_size),
            "disease_accuracy": float(disease_metrics["accuracy"]),
            "disease_macro_recall": macro_recall,
            "disease_macro_f1": macro_f1,
            "class_metrics": class_report,
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
        saved_model_names = [
            "disease_model.keras",
            "crop_type_model.keras",
            "class_indices.json",
            "training_metrics.json",
        ]
        try:
            model.save(os.path.join(staging_dir, "disease_model.h5"))
            saved_model_names.insert(1, "disease_model.h5")
        except Exception as error:
            print(f"[!] H5 export skipped: {error}")
        crop_model.save(os.path.join(staging_dir, "crop_type_model.keras"))
        with open(os.path.join(staging_dir, "class_indices.json"), "w", encoding="utf-8") as metadata_file:
            json.dump(metadata_export, metadata_file, indent=2)
        with open(os.path.join(staging_dir, "training_metrics.json"), "w", encoding="utf-8") as metrics_file:
            json.dump(metrics_export, metrics_file, indent=2)

        tflite_models = export_tflite_models(model, crop_model, android_assets_dir, img_size)
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

        for file_name in saved_model_names:
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
    print(f"[+] Held-out macro recall: {metrics_export['disease_macro_recall']:.4f}")
    print(f"[+] Held-out macro F1: {metrics_export['disease_macro_f1']:.4f}")
    print(f"[+] Held-out crop accuracy: {metrics_export['crop_accuracy']:.4f}")
    print("[+] Keras, H5, metadata, training metrics, and Android TFLite models were updated.")

def _leaf_array(path, cache):
    cached = cache.get(path)
    if cached is not None:
        return cached
    image = Image.open(path).convert("RGB")
    image.thumbnail((360, 360))
    array = np.asarray(image, dtype=np.uint8)
    cache[path] = array
    if len(cache) > 500:
        cache.popitem(last=False)
    return array


def _paste_leaf(canvas, leaf, rng, scale_range):
    side = int(canvas.shape[0] * rng.uniform(*scale_range))
    side = int(np.clip(side, 64, canvas.shape[0] - 8))
    resized = np.asarray(Image.fromarray(leaf).resize((side, side), Image.Resampling.BILINEAR))
    saturation = resized.max(axis=2).astype(np.int16) - resized.min(axis=2).astype(np.int16)
    mask = saturation > 20
    top = int(rng.integers(0, canvas.shape[0] - side + 1))
    left = int(rng.integers(0, canvas.shape[1] - side + 1))
    region = canvas[top:top + side, left:left + side]
    region[mask] = resized[mask]


def _whole_plant_canvas(rng, paths_by_label, label, cache):
    class_name = POTATO_TOMATO_CLASSES[label]
    crop_prefix = "Potato" if class_name.startswith("Potato") else "Tomato"
    healthy_label = next(
        index for index, name in enumerate(POTATO_TOMATO_CLASSES)
        if name.startswith(crop_prefix) and "healthy" in name.lower()
    )
    background = _leaf_array(paths_by_label[healthy_label][int(rng.integers(len(paths_by_label[healthy_label])))], cache)
    canvas_image = Image.fromarray(background).resize((480, 480), Image.Resampling.BILINEAR)
    canvas_image = canvas_image.filter(ImageFilter.GaussianBlur(radius=7))
    canvas = np.asarray(canvas_image, dtype=np.float32) * float(rng.uniform(0.72, 1.08))
    shade = np.linspace(0.78, 1.08, 480, dtype=np.float32)
    if rng.random() < 0.5:
        shade = shade[:, None]
    else:
        shade = shade[None, :]
    canvas *= shade[..., None]
    for _ in range(int(rng.integers(2, 5))):
        filler = _leaf_array(paths_by_label[healthy_label][int(rng.integers(len(paths_by_label[healthy_label])))], cache)
        _paste_leaf(canvas, filler, rng, (0.25, 0.55))
    if "healthy" not in class_name.lower():
        diseased = _leaf_array(paths_by_label[label][int(rng.integers(len(paths_by_label[label])))], cache)
        _paste_leaf(canvas, diseased, rng, (0.42, 0.78))
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(canvas).resize((160, 160), Image.Resampling.BILINEAR))


def adapt_whole_plant(
    dataset_dir=None,
    epochs=4,
    batch_size=32,
    output_dir="model",
    android_assets_dir=os.path.join("android", "app", "src", "main", "assets"),
):
    """Teach the saved model to read a frame that contains several leaves."""
    train_path, val_path = resolve_dataset_paths(dataset_dir)
    tf.keras.utils.set_random_seed(7)
    train_examples, validation_examples, train_counts, validation_counts, duplicate_count = (
        create_stratified_splits([train_path], [val_path] if val_path else None)
    )
    paths_by_label = {index: [] for index in range(len(POTATO_TOMATO_CLASSES))}
    for path, label in train_examples:
        paths_by_label[label].append(path)
    img_size = (160, 160)
    val_ds = create_dataset(validation_examples, batch_size, img_size, shuffle=False, seed=7, training=False)
    model = tf.keras.models.load_model(os.path.join(output_dir, "disease_model.keras"), compile=False)
    compile_disease_model(model, learning_rate=1e-5)
    baseline_weights = model.get_weights()
    baseline = model.evaluate(val_ds, verbose=0, return_dict=True)
    baseline_accuracy = float(baseline["accuracy"])
    print(f"Baseline held-out accuracy: {baseline_accuracy:.4f}")

    rng = np.random.default_rng(7)
    cache = OrderedDict()
    images = []
    labels = []
    for label in range(len(POTATO_TOMATO_CLASSES)):
        for _ in range(160):
            images.append(_whole_plant_canvas(rng, paths_by_label, label, cache))
            labels.append(label)
        rehearsal_paths = paths_by_label[label][:]
        rng.shuffle(rehearsal_paths)
        for path in rehearsal_paths[:70]:
            leaf = _leaf_array(path, cache)
            framed = np.asarray(
                Image.fromarray(leaf).resize((160, 160), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            images.append(framed)
            labels.append(label)
    images = np.stack(images).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    print(f"Whole-plant adaptation images: {len(labels)}")

    unfreeze_backbone_top(model, trainable_layers=30)
    compile_disease_model(model, learning_rate=2e-5)
    class_weights = disease_class_weights([int(np.sum(labels == index)) for index in range(len(POTATO_TOMATO_CLASSES))])
    history = model.fit(
        images,
        labels,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)],
        verbose=2,
    )
    adapted = model.evaluate(val_ds, verbose=0, return_dict=True)
    adapted_accuracy = float(adapted["accuracy"])
    print(f"Adapted held-out accuracy: {adapted_accuracy:.4f}")
    if adapted_accuracy + 0.03 < baseline_accuracy:
        model.set_weights(baseline_weights)
        print("Adaptation lowered the held-out score too far, so the previous model was kept.")
        return

    disease_probabilities = model.predict(val_ds, verbose=0)
    disease_predictions = np.argmax(disease_probabilities, axis=1)
    validation_labels = np.asarray([label for _, label in validation_examples], dtype=np.int32)
    class_report, macro_recall, macro_f1 = classification_scores(
        validation_labels, disease_predictions, POTATO_TOMATO_CLASSES
    )
    previous_metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(previous_metrics_path, "r", encoding="utf-8") as metrics_file:
        metrics_export = json.load(metrics_file)
    metrics_export.update({
        "architecture": "MobileNetV2 fine-tuned on single leaves and multi-leaf plant scenes",
        "disease_accuracy": adapted_accuracy,
        "disease_macro_recall": macro_recall,
        "disease_macro_f1": macro_f1,
        "class_metrics": class_report,
        "class_validation_counts": dict(zip(POTATO_TOMATO_CLASSES, validation_counts)),
        "whole_plant_adaptation_images": int(len(labels)),
        "baseline_disease_accuracy": baseline_accuracy,
    })
    with open(os.path.join(output_dir, "class_indices.json"), "r", encoding="utf-8") as metadata_file:
        metadata_export = json.load(metadata_file)
    model.save(os.path.join(output_dir, "disease_model.keras"))
    try:
        model.save(os.path.join(output_dir, "disease_model.h5"))
    except Exception as error:
        print(f"[!] H5 export skipped: {error}")
    with open(previous_metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics_export, metrics_file, indent=2)
    crop_model = tf.keras.models.load_model(os.path.join(output_dir, "crop_type_model.keras"), compile=False)
    tflite_models = export_tflite_models(model, crop_model, android_assets_dir, img_size)
    for file_name, model_bytes in tflite_models.items():
        with open(os.path.join(android_assets_dir, file_name), "wb") as model_file:
            model_file.write(model_bytes)
    for file_name, payload in (
        ("class_indices.json", metadata_export),
        ("training_metrics.json", metrics_export),
    ):
        with open(os.path.join(android_assets_dir, file_name), "w", encoding="utf-8") as metadata_file:
            json.dump(payload, metadata_file, indent=2)
    best_epoch = max(history.history.get("val_accuracy", [adapted_accuracy]))
    print(f"[+] Best adaptation validation accuracy this run: {best_epoch:.4f}")
    print("[+] Disease model updated for multi-leaf photos. The crop model was left as it was.")


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
    parser.add_argument("--epochs", type=int, default=8, help="Epochs for the frozen-backbone phase")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=160, help="Square MobileNetV2 input resolution")
    parser.add_argument(
        "--adapt-whole-plant",
        action="store_true",
        help="Fine-tune the saved disease model on multi-leaf scenes",
    )
    args = parser.parse_args()

    if args.adapt_whole_plant:
        adapt_whole_plant(
            dataset_dir=args.dataset_path,
            epochs=args.epochs,
        )
    else:
        train(
        dataset_dir=args.dataset_path,
        additional_dataset_dirs=[] if args.skip_supplemental_dataset else args.additional_dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size)
    )
