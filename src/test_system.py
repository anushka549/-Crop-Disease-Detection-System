"""Integration checks for paths, labels, model, preprocessing, and lookup."""
from pathlib import Path
import sys
import tensorflow as tf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import IMAGE_SIZE
from disease_info import DISEASE_INFO, get_disease_info
from predict import preprocess_image

def check(label, function):
    try: function(); print(f"[OK] {label}")
    except Exception as error: print(f"[ERROR] {label}: {error}")
def main():
    def dataset_check():
        names = []
        for split in ("train", "validation", "test"):
            path = ROOT / "data" / split
            if not path.is_dir(): raise FileNotFoundError(f"Missing data/{split}")
            names.append(sorted(p.name for p in path.iterdir() if p.is_dir()))
        if not names[0] or names[0] != names[1] or names[0] != names[2]: raise ValueError("Split class names do not match")
        if len(names[0]) != 15: raise ValueError(f"Expected 15 classes, found {len(names[0])}")
    check("dataset paths and 15 matching classes", dataset_check)
    check("disease information has 15 entries", lambda: None if len(DISEASE_INFO) == 15 else (_ for _ in ()).throw(ValueError(f"found {len(DISEASE_INFO)}")))
    check("disease information lookup", lambda: None if get_disease_info("Tomato_healthy") and not get_disease_info("unknown") else (_ for _ in ()).throw(ValueError("lookup mismatch")))
    image = next((ROOT / "data" / "test").glob("*/*.jpg"), None)
    if image:
        check("image preprocessing", lambda: None if preprocess_image(image).shape == (1, *IMAGE_SIZE, 3) else (_ for _ in ()).throw(ValueError("unexpected shape")))
    else: print("[SKIP] preprocessing: no JPG found")
    model_path = ROOT / "models" / "final_crop_disease_model.keras"
    if not model_path.exists(): model_path = ROOT / "models" / "best_crop_disease_model.keras"
    if model_path.exists():
        def model_check():
            model = tf.keras.models.load_model(model_path)
            if image:
                probs = model.predict(preprocess_image(image), verbose=0)[0]
                if len(probs) != 15 or abs(float(probs.sum()) - 1) > 0.02: raise ValueError("invalid prediction output")
        check("model loading and prediction", model_check)
    else: print("[SKIP] model loading/prediction: train model first")
if __name__ == "__main__": main()
