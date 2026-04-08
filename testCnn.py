from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


SEED = 42
DATA_DIR = Path("data")
MODEL_PATH = Path("model") / "flower_cnn_best.h5"
CLASS_NAMES_PATH = Path("model") / "class_names.json"
CONFUSION_MATRIX_PATH = Path("confusion_matrix.png")
NORMALIZED_CONFUSION_MATRIX_PATH = Path("confusion_matrix_normalized.png")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
VALID_EXTENSIONS = {".jpg", ".jpeg"}
AUTOTUNE = tf.data.AUTOTUNE
TEST_SIZE = 0.2


def configure_environment() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available for inference: {[gpu.name for gpu in gpus]}")
    else:
        print("GPU not found. Inference will run on CPU.")


def collect_image_paths(data_dir: Path) -> tuple[list[str], list[int], list[str]]:
    image_paths: list[str] = []
    labels: list[int] = []
    class_names: list[str] = []
    skipped_files: list[str] = []

    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    for label, class_dir in enumerate(class_dirs):
        valid_files: list[Path] = []
        for file in sorted(class_dir.iterdir()):
            if not file.is_file() or file.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if is_supported_image(file):
                valid_files.append(file)
            else:
                skipped_files.append(str(file))

        if not valid_files:
            continue
        class_names.append(class_dir.name)
        image_paths.extend(str(file) for file in valid_files)
        labels.extend([label] * len(valid_files))

    if not image_paths:
        raise SystemExit("No JPG/JPEG files were found under data/.")

    if skipped_files:
        print(f"Skipped {len(skipped_files)} invalid JPG/JPEG file(s).")
        for path in skipped_files:
            print(f"  - {path}")

    return image_paths, labels, class_names


def is_supported_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def decode_and_resize(image_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_dataset(image_paths: list[str], labels: list[int]) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


def load_class_names(default_class_names: list[str]) -> list[str]:
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return default_class_names


def save_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()


def save_normalized_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    cm_normalized = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(NORMALIZED_CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()


def main() -> None:
    configure_environment()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")

    image_paths, labels, default_class_names = collect_image_paths(DATA_DIR)
    _, test_paths, _, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    class_names = load_class_names(default_class_names)
    test_dataset = build_dataset(test_paths, test_labels)

    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.array(test_labels)

    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Test accuracy: {accuracy:.4f}")

    cm = confusion_matrix(true_labels, predicted_labels)
    save_confusion_matrix(cm, class_names)
    save_normalized_confusion_matrix(cm, class_names)
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")
    print(f"Saved normalized confusion matrix to: {NORMALIZED_CONFUSION_MATRIX_PATH}")

    report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)
    print("Classification report:")
    print(report)


if __name__ == "__main__":
    main()
