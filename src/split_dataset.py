import os
import shutil
import random

SOURCE = "data/raw/PlantVillage"

TRAIN = "data/train"
VALIDATION = "data/validation"
TEST = "data/test"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# Create folders
os.makedirs(TRAIN, exist_ok=True)
os.makedirs(VALIDATION, exist_ok=True)
os.makedirs(TEST, exist_ok=True)

classes = [
    folder for folder in os.listdir(SOURCE)
    if os.path.isdir(os.path.join(SOURCE, folder))
    and folder != "PlantVillage"
]

print("Creating dataset split...\n")

for class_name in sorted(classes):

    source_folder = os.path.join(SOURCE, class_name)

    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    # Create class folders
    train_folder = os.path.join(TRAIN, class_name)
    validation_folder = os.path.join(VALIDATION, class_name)
    test_folder = os.path.join(TEST, class_name)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Copy images
    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(train_folder, image)
        )

    for image in validation_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(validation_folder, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(test_folder, image)
        )

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(validation_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset splitting completed successfully!")