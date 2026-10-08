"""Streamlit demonstration interface for crop disease classification."""
from pathlib import Path
import json, sys
from datetime import datetime
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from disease_info import get_disease_info
MODEL_DIR = ROOT / "models"
HISTORY_PATH = ROOT / "results" / "training_history.json"
PREDICTION_HISTORY = ROOT / "results" / "prediction_history.json"
IMAGE_SIZE = (224, 224)
LOW_CONFIDENCE = 0.55
st.set_page_config(page_title="Crop Disease Detection System", page_icon="🌱", layout="wide")
st.title("🌱 Crop Disease Detection System")
st.markdown("A transfer-learning demonstration that classifies leaf images into 15 PlantVillage crop-health categories. Predictions are educational and should be confirmed by a qualified local agricultural professional.")
if not (MODEL_DIR / "final_crop_disease_model.keras").exists() and not (MODEL_DIR / "best_crop_disease_model.keras").exists():
    st.warning("No trained model is available yet. Run python src/train.py before making predictions.")
@st.cache_resource
def load_model_once(model_path_text):
    return tf.keras.models.load_model(model_path_text)
def model_and_labels():
    final = MODEL_DIR / "final_crop_disease_model.keras"
    best = MODEL_DIR / "best_crop_disease_model.keras"
    selected = final if final.exists() else best
    if not selected.exists(): return None, None, None
    labels = []
    if HISTORY_PATH.exists():
        try: labels = json.loads(HISTORY_PATH.read_text(encoding="utf-8")).get("class_names", [])
        except (OSError, json.JSONDecodeError): pass
    return load_model_once(str(selected)), labels, selected
def predict_image(image, model, labels):
    rgb = image.convert("RGB").resize(IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(rgb)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    probabilities = model.predict(np.expand_dims(array, 0), verbose=0)[0]
    if len(labels) != len(probabilities): raise ValueError("Model labels are unavailable or do not match model outputs. Retrain to generate training_history.json.")
    order = np.argsort(probabilities)[::-1][:3]
    return [{"class_name": labels[int(i)], "probability": float(probabilities[i])} for i in order]
def record_prediction(item):
    PREDICTION_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if PREDICTION_HISTORY.exists():
        try: records = json.loads(PREDICTION_HISTORY.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): pass
    records.insert(0, item)
    PREDICTION_HISTORY.write_text(json.dumps(records[:20], indent=2), encoding="utf-8")
with st.sidebar:
    st.header("About this demo")
    st.write("Supported crops: bell pepper, potato, and tomato.")
    st.write("The model recognizes 15 healthy/disease categories from PlantVillage.")
    st.caption("Field photos and unfamiliar conditions may reduce reliability.")
home_tab, predict_tab, history_tab = st.tabs(["Home", "Image prediction", "Recent predictions"])
with home_tab:
    st.subheader("Project overview")
    st.write("This capstone uses MobileNetV2 to provide a screening label for selected crop leaf conditions.")
    st.write("Workflow: upload a leaf image → resize and preprocess → classify → review general educational information.")
    st.info("Train a model first with python src/train.py. The app never starts training automatically.")
    with st.expander("Supported labels"):
        if HISTORY_PATH.exists():
            try:
                for name in json.loads(HISTORY_PATH.read_text(encoding="utf-8")).get("class_names", []): st.write("• " + name.replace("___", " — ").replace("__", " — ").replace("_", " "))
            except (OSError, json.JSONDecodeError): st.write("Label list not available yet; see src/disease_info.py.")
        else: st.write("See the 15 labels in src/disease_info.py.")
with predict_tab:
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded.getvalue()).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError) as error: st.error(f"Could not open image. Choose a valid JPG, JPEG, or PNG. Details: {error}")
    if st.button("Predict disease", type="primary", disabled=image is None):
        try:
            model, labels, model_path = model_and_labels()
            if model is None: st.error("No trained model found. Run python src/train.py, then refresh this app.")
            else:
                with st.spinner("Analyzing the image..."): predictions = predict_image(image, model, labels)
                top = predictions[0]
                info = get_disease_info(top["class_name"])
                st.subheader("Prediction result")
                if top["probability"] < LOW_CONFIDENCE: st.warning(f"Low confidence ({top['probability']*100:.1f}%). Treat this as uncertain and consider expert review.")
                else: st.success(f"Confidence: {top['probability']*100:.1f}%")
                if info:
                    st.write(f"**Crop:** {info['crop']}")
                    if info.get("status") == "Healthy": st.write("**Status:** Healthy"); st.write(info["care"])
                    else:
                        st.write(f"**Disease:** {info['disease']}")
                        st.write(f"**Symptoms:** {info['symptoms']}")
                        st.write(f"**Prevention:** {info['prevention']}")
                        st.write(f"**General management:** {info['management']}")
                else: st.warning("No curated information is available for this label.")
                st.markdown("**Top 3 predictions**")
                for result in predictions: st.write(f"{result['class_name']} — {result['probability']*100:.2f}%")
                record_prediction({"time": datetime.now().astimezone().isoformat(timespec="seconds"), "class_name": top["class_name"], "confidence": top["probability"], "model": model_path.name})
        except (OSError, ValueError, tf.errors.InvalidArgumentError) as error: st.error(f"Prediction could not be completed: {error}")
with history_tab:
    st.subheader("Recent predictions on this device")
    if PREDICTION_HISTORY.exists():
        try:
            records = json.loads(PREDICTION_HISTORY.read_text(encoding="utf-8"))
            if records: st.dataframe([{"Time": r.get("time"), "Prediction": r.get("class_name"), "Confidence": f"{r.get('confidence',0)*100:.1f}%", "Model": r.get("model")} for r in records], use_container_width=True, hide_index=True)
            else: st.caption("No predictions recorded yet.")
        except (OSError, json.JSONDecodeError): st.warning("Prediction history could not be read.")
    else: st.caption("Recent predictions appear here after the first successful prediction.")
