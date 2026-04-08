from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
BEST_MODEL_PATH = MODEL_DIR / "flower_cnn_best.h5"
FINAL_MODEL_PATH = MODEL_DIR / "flower_cnn_final.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
TRAINING_CURVES_PATH = Path("training_curves.png")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE
VALID_EXTENSIONS = {".jpg", ".jpeg"}
TEST_SIZE = 0.2
VALIDATION_SIZE_FROM_TRAIN = 0.2


def configure_environment() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {[gpu.name for gpu in gpus]}")
    else:
        print("GPU not found. Training will run on CPU.")


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


def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE[0] + 12, IMAGE_SIZE[1] + 12)
    image = tf.image.random_crop(image, size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_dataset(
    image_paths: list[str],
    labels: list[int],
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        dataset = dataset.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_class_weights(labels: list[int]) -> dict[int, float]:
    label_counts = np.bincount(labels)
    total = len(labels)
    num_classes = len(label_counts)
    return {
        class_index: total / (num_classes * count)
        for class_index, count in enumerate(label_counts)
        if count > 0
    }


def save_training_curves(history: tf.keras.callbacks.History) -> None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_CURVES_PATH, dpi=200)
    plt.close()


def main() -> None:
    configure_environment()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    image_paths, labels, class_names = collect_image_paths(DATA_DIR)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=VALIDATION_SIZE_FROM_TRAIN,
        random_state=SEED,
        stratify=train_val_labels,
    )

    train_dataset = build_dataset(train_paths, train_labels, training=True)
    val_dataset = build_dataset(val_paths, val_labels, training=False)
    test_dataset = build_dataset(test_paths, test_labels, training=False)

    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")

    model = build_model(num_classes=len(class_names))
    model.summary()
    class_weights = build_class_weights(train_labels)
    print(f"Class weights: {class_weights}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=4,
            mode="max",
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    best_model.save(FINAL_MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    save_training_curves(history)

    final_loss, final_accuracy = best_model.evaluate(test_dataset, verbose=0)
    print(f"Final test loss: {final_loss:.4f}")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    print(f"Saved best model to: {BEST_MODEL_PATH}")
    print(f"Saved final model to: {FINAL_MODEL_PATH}")
    print(f"Saved class names to: {CLASS_NAMES_PATH}")
    print(f"Saved training curves to: {TRAINING_CURVES_PATH}")
    print(f"Training history keys: {list(history.history.keys())}")


if __name__ == "__main__":
    main()
