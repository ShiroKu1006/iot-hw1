from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("model") / "flower_cnn_final.h5"
CLASS_NAMES_PATH = Path("model") / "class_names.json"
IMAGE_SIZE = (160, 160)


def load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        raise SystemExit(f"Class names file not found: {CLASS_NAMES_PATH}")
    return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))


def preprocess_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32) / 255.0

    return np.expand_dims(image_array, axis=0)


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")

    class_names = load_class_names()
    image_batch = preprocess_image(IMAGE_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(image_batch, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    print(f"Image path: {IMAGE_PATH}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("All class probabilities:")

    for class_name, probability in zip(class_names, predictions):
        print(f"  {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
