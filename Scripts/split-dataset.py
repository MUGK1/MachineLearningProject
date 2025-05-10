import os
import shutil
import random

# Define paths
input_dir = "../Dataset-Resized"  # Use your resized dataset path
output_dir = "../Dataset-Split"  # Folder where split dataset will be stored

# Split ratio
train_ratio = 0.8  # 80% training, 20% testing

# Create directories
for split in ["train", "test"]:
    for class_name in os.listdir(input_dir):
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# Split and move files
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    train_split = int(len(images) * train_ratio)

    for i, img_name in enumerate(images):
        img_path = os.path.join(class_path, img_name)

        if i < train_split:
            dest = os.path.join(output_dir, "train", class_name, img_name)
        else:
            dest = os.path.join(output_dir, "test", class_name, img_name)

        shutil.copy(img_path, dest)

print("Dataset split complete!")
