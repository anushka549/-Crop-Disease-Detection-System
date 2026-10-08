"""Optional MobileNetV2 fine-tuning after successful initial training."""
from pathlib import Path
import tensorflow as tf
from data_loader import load_datasets, PROJECT_ROOT


def main(epochs=5, unfreeze_last=30):
    models_dir = PROJECT_ROOT / "models"
    initial = models_dir / "best_crop_disease_model.keras"
    if not initial.exists():
        raise FileNotFoundError("Initial best model is missing; train successfully before fine-tuning.")
    train_ds, val_ds, _, _, _ = load_datasets(verbose=False)
    model = tf.keras.models.load_model(initial)
    base = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower()), None)
    if base is None:
        raise RuntimeError("Could not locate the MobileNetV2 base model in the saved model")
    base.trainable = True
    cutoff = max(0, len(base.layers) - unfreeze_last)
    for layer in base.layers[:cutoff]:
        layer.trainable = False
    for layer in base.layers[cutoff:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    target = models_dir / "final_crop_disease_model.keras"
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
                 tf.keras.callbacks.ModelCheckpoint(target, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print(f"Fine-tuning complete. Best observed validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print("Run python src/evaluate.py to compare held-out test performance with the initial model report.")


if __name__ == "__main__":
    main()
