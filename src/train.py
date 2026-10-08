"""Train the initial crop disease classifier from the project root."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_datasets, PROJECT_ROOT
from model import build_model

EPOCHS = 12
BATCH_SIZE = 32


def main():
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    train_ds, val_ds, _, class_names, counts = load_datasets(batch_size=BATCH_SIZE)
    model = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(models_dir / "best_crop_disease_model.keras", monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
    history_data.update({"class_names": class_names, "image_counts": counts})
    (results_dir / "training_history.json").write_text(json.dumps(history_data, indent=2), encoding="utf-8")
    epochs = range(1, len(history.history["loss"]) + 1)
    for metric, title, filename in (("accuracy", "Accuracy", "accuracy.png"), ("loss", "Loss", "loss.png")):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history.history[metric], label=f"Training {metric}")
        plt.plot(epochs, history.history[f"val_{metric}"], label=f"Validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"Training and validation {title.lower()}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(results_dir / filename, dpi=150)
        plt.close()
    if not (models_dir / "best_crop_disease_model.keras").is_file():
        raise RuntimeError("Training ended without creating the best model checkpoint")
    print(f"Best model saved to: {models_dir / 'best_crop_disease_model.keras'}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
