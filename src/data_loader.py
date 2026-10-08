"""Load the prepared PlantVillage split for MobileNetV2."""
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def _count_images(folder: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def load_datasets(batch_size: int = DEFAULT_BATCH_SIZE, image_size=IMAGE_SIZE, seed: int = SEED, verbose: bool = True):
    """Return (train, validation, test, class_names, image_counts).

    Images are decoded and resized by Keras, augmented only in the training
    pipeline, then scaled to the [-1, 1] range expected by MobileNetV2.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    paths = {name: DATA_ROOT / name for name in ("train", "validation", "test")}
    for name, path in paths.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {name} dataset folder: {path}")

    train = tf.keras.utils.image_dataset_from_directory(
        paths["train"], image_size=image_size, batch_size=batch_size,
        label_mode="int", shuffle=True, seed=seed)
    class_names = list(train.class_names)
    datasets = {"train": train}
    for name in ("validation", "test"):
        ds = tf.keras.utils.image_dataset_from_directory(
            paths[name], image_size=image_size, batch_size=batch_size,
            label_mode="int", shuffle=False)
        if list(ds.class_names) != class_names:
            raise ValueError(f"Class folders in {name} do not match training classes")
        datasets[name] = ds

    augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.08, seed=seed + 1),
        tf.keras.layers.RandomZoom(0.10, seed=seed + 2),
        tf.keras.layers.RandomContrast(0.10, seed=seed + 3),
    ], name="training_augmentation")

    def prepare(images, labels, training=False):
        images = tf.cast(images, tf.float32)
        if training:
            images = augmenter(images, training=True)
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    prepared = {
        "train": datasets["train"].map(lambda x, y: prepare(x, y, True), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE),
        "validation": datasets["validation"].map(lambda x, y: prepare(x, y, False), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE),
        "test": datasets["test"].map(lambda x, y: prepare(x, y, False), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE),
    }
    counts = {name: _count_images(path) for name, path in paths.items()}
    if verbose:
        print(f"Number of classes: {len(class_names)}")
        print("Class names:", ", ".join(class_names))
        print(f"Training images: {counts['train']:,}")
        print(f"Validation images: {counts['validation']:,}")
        print(f"Test images: {counts['test']:,}")
        print(f"Image shape: {image_size[0]} x {image_size[1]} x 3")
    return prepared["train"], prepared["validation"], prepared["test"], class_names, counts


if __name__ == "__main__":
    load_datasets()
