import os

DATASET_PATH = "data/raw/PlantVillage"

print("Checking Crop Disease Dataset...\n")

classes = []

for name in os.listdir(DATASET_PATH):
    path = os.path.join(DATASET_PATH, name)

    if os.path.isdir(path) and name != "PlantVillage":
        classes.append(name)

print("Total classes:", len(classes))
print("\nAvailable classes:\n")

for i, class_name in enumerate(sorted(classes), 1):

    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"{i}. {class_name} -> {len(images)} images")

print("\nDataset check completed!")