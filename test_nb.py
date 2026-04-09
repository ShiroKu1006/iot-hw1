from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_PATH = Path("data/(4) 朱槿 Hibiscus rosa-sinensis/0048_obs346482119_photo631589443.jpg")
MODEL_CANDIDATES = [
    Path("model") / "network_binary.nb",
    Path("model") / "img_class_cnn.nb",
]
CLASS_NAMES_PATH = Path("model") / "class_names.json"
IMAGE_SIZE = (160, 160)


def load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        raise SystemExit(f"Class names file not found: {CLASS_NAMES_PATH}")
    return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))


def resolve_model_path() -> Path:
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return model_path

    checked_paths = "\n".join(f"  - {path}" for path in MODEL_CANDIDATES)
    raise SystemExit(f"NB model file not found. Checked these paths:\n{checked_paths}")


def preprocess_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.uint8)

    # The NB model header suggests an rgb_uint8_NCHW input layout.
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0)


def load_external_adapter() -> Any:
    try:
        return importlib.import_module("nb_runtime_adapter")
    except ModuleNotFoundError:
        return None


def run_nb_inference(model_path: Path, input_tensor: np.ndarray) -> np.ndarray:
    adapter = load_external_adapter()
    if adapter is None:
        raise SystemExit(
            "Cannot run .nb inference yet because no NB runtime adapter was found.\n"
            f"Detected NB model: {model_path}\n"
            "This project needs a vendor-specific runtime for this `.nb` model.\n"
            "Please add a file named `nb_runtime_adapter.py` with a function:\n"
            "    predict(model_path, input_tensor) -> numpy.ndarray\n"
            "Expected input_tensor format in this script is: uint8, shape=(1, 3, 160, 160).\n"
            "If you have the teacher's SDK or demo code, send it to me and I can wire it in."
        )

    if not hasattr(adapter, "predict"):
        raise SystemExit(
            "`nb_runtime_adapter.py` was found, but it does not define "
            "`predict(model_path, input_tensor)`."
        )

    output = adapter.predict(model_path, input_tensor)
    output_array = np.asarray(output)

    if output_array.ndim == 2 and output_array.shape[0] == 1:
        output_array = output_array[0]
    elif output_array.ndim != 1:
        raise SystemExit(
            "The adapter returned an unsupported output shape: "
            f"{output_array.shape}. Expected (num_classes,) or (1, num_classes)."
        )

    return output_array.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def main() -> None:
    model_path = resolve_model_path()
    class_names = load_class_names()
    input_tensor = preprocess_image(IMAGE_PATH)
    raw_output = run_nb_inference(model_path, input_tensor)

    if raw_output.size != len(class_names):
        raise SystemExit(
            "The model output size does not match the number of classes.\n"
            f"Output size: {raw_output.size}\n"
            f"Class count: {len(class_names)}"
        )

    probabilities = softmax(raw_output)
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    print(f"Image path: {IMAGE_PATH}")
    print(f"Model path: {model_path}")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("All class probabilities:")

    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
