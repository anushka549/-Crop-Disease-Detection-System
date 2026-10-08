"""Predict the most likely crop disease for one image."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from data_loader import PROJECT_ROOT
from data_loader import IMAGE_SIZE


def preprocess_image(image_path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = tf.keras.utils.load_img(path, target_size=IMAGE_SIZE, color_mode="rgb")
    array = tf.keras.utils.img_to_array(image)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Classify a crop leaf image")
    parser.add_argument("image_path", help="Path to a JPG, JPEG, or PNG image")
    args = parser.parse_args()
    final = PROJECT_ROOT / "models" / "final_crop_disease_model.keras"
    best = PROJECT_ROOT / "models" / "best_crop_disease_model.keras"
    model_path = final if final.exists() else best
    if not model_path.exists():
        parser.error("No trained model found. Run python src/train.py first.")
    try:
        image = preprocess_image(args.image_path)
        model = tf.keras.models.load_model(model_path)
        probs = model.predict(image, verbose=0)[0]
        names_path = PROJECT_ROOT / "results" / "training_history.json"
        import json
        class_names = json.loads(names_path.read_text(encoding="utf-8")).get("class_names") if names_path.exists() else None
        if not class_names or len(class_names) != len(probs):
            class_names = [str(i) for i in range(len(probs))]
        indices = np.argsort(probs)[::-1][:3]
        print(f"Predicted class: {class_names[indices[0]]}")
        print(f"Confidence: {probs[indices[0]] * 100:.2f}%")
        print("Top 3 predictions:")
        for index in indices:
            print(f"  {class_names[index]}: {probs[index] * 100:.2f}%")
    except (OSError, ValueError, tf.errors.InvalidArgumentError) as error:
        parser.error(f"Could not process image: {error}")


if __name__ == "__main__":
    main()
