# Crop Disease Detection System

A final-year capstone demonstration that classifies leaf images for bell pepper, potato, and tomato into 15 PlantVillage categories using MobileNetV2 transfer learning. It includes data loading, training, test evaluation, single-image prediction, general disease information, and a Streamlit interface.

## Problem statement
Visible crop symptoms can be difficult to identify consistently. This project explores an image-based screening workflow that maps a leaf photo to one of the supported PlantVillage labels. It is educational and does not replace diagnosis by an agricultural expert.

## Objectives
- Reuse the prepared PlantVillage train, validation, and test folders.
- Build a TensorFlow pipeline with training-only augmentation.
- Train a transfer-learning classifier and report held-out metrics.
- Provide CLI and browser-based prediction with general crop-health information.

## Features
- MobileNetV2 ImageNet transfer learning, initially frozen.
- 224 × 224 RGB images; configurable batch size (default 32).
- Training history, accuracy/loss graphs, weighted test metrics and confusion matrix.
- Optional fine-tuning of the final MobileNetV2 layers.
- Prediction top-three results and recent local prediction history in Streamlit.

## Dataset and 15 classes
Uses the existing PlantVillage dataset under data/raw/PlantVillage and prepared folders data/train, data/validation, and data/test. The scripts do not download a dataset. The current split contains 14,440 training images, 3,089 validation images, and 3,109 test images (20,638 total). This is below the approximate 22,638 noted in the project brief, so the existing split should be reconciled with the raw dataset before interpreting metrics.

1. Pepper__bell___Bacterial_spot
2. Pepper__bell___healthy
3. Potato___Early_blight
4. Potato___Late_blight
5. Potato___healthy
6. Tomato_Bacterial_spot
7. Tomato_Early_blight
8. Tomato_Late_blight
9. Tomato_Leaf_Mold
10. Tomato_Septoria_leaf_spot
11. Tomato_Spider_mites_Two_spotted_spider_mite
12. Tomato__Target_Spot
13. Tomato__Tomato_YellowLeaf__Curl_Virus
14. Tomato__Tomato_mosaic_virus
15. Tomato_healthy

## Technologies
Python 3.12, TensorFlow/Keras 2.21, MobileNetV2, NumPy, Matplotlib, scikit-learn, Pillow, and Streamlit.

## System architecture
PlantVillage split folders → src/data_loader.py (resize, augmentation only for train, MobileNetV2 preprocessing) → src/model.py (frozen MobileNetV2 + classifier) → src/train.py → models/best_crop_disease_model.keras → evaluation, CLI prediction, or Streamlit app.

## Data preprocessing
Keras loads class-folder labels at 224 × 224. Training batches use random horizontal flips, small rotations, zoom, and contrast. Validation and test are not augmented. All pipelines use MobileNetV2 preprocess_input to scale pixels into the expected range. Dataset shuffling and augmentation use a fixed seed.

## Model architecture
ImageNet-pretrained MobileNetV2 with include_top=False and a frozen base initially; GlobalAveragePooling2D; 256-unit ReLU Dense layer; Dropout 0.35; configurable softmax output (15 classes for this dataset). The model uses Adam and sparse categorical cross-entropy. The pipeline supports CPU; training time depends on the machine.

## Training process
Training defaults to 12 epochs and batch size 32, with EarlyStopping, best validation accuracy ModelCheckpoint, and ReduceLROnPlateau. Outputs are models/best_crop_disease_model.keras, results/training_history.json, results/accuracy.png, and results/loss.png. The first ImageNet weights download may need network access if not cached.

## Evaluation metrics
Evaluation uses only the held-out test split and reports test loss, accuracy, weighted precision, weighted recall, weighted F1, and per-class precision/recall/F1. It writes results/classification_report.txt and results/confusion_matrix.png. Run evaluation after fine-tuning to compare. No accuracy is claimed before an actual successful run.

## How to run
Run commands from the project root in PowerShell:

~~~powershell
.venv\Scripts\activate
python src\check_dataset.py
python src\data_loader.py
python src\train.py
python src\evaluate.py
python src\predict.py "path/to/leaf.jpg"
streamlit run app\app.py
~~~

Optional fine-tuning and integration checks:

~~~powershell
python src\fine_tune.py
python src\evaluate.py
python src\test_system.py
~~~

## Expected output
Training saves a best checkpoint and plots. Evaluation prints test metrics and class-wise results. Prediction prints the most likely class, confidence percentage, and top three labels. The web app shows curated crop/disease information where available and records up to 20 recent predictions at results/prediction_history.json. The Streamlit app never trains on startup.

## Limitations
The classifier recognizes only these 15 labels and learns from PlantVillage images. Field lighting, backgrounds, mixed symptoms, other crops, nutrient problems, and local varieties may differ. Softmax confidence is not a calibrated probability of correctness. Management notes are general education, not chemical instructions; consult local agricultural extension and follow local regulations.

## Future enhancements
Evaluate on independently collected field images; assess confidence calibration; add verified labels for more crops; explore mobile/edge deployment; add expert-reviewed regional guidance and uncertainty estimation.
