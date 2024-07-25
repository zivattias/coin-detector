"""
This module is responsible for splitting the dataset into train, validation, and test sets.
It assumes that the dataset is organized in a specific structure, with JPG images and TXT labels, in YOLO format.
Corresponding images and labels must have the same name, e.g. image 'foo.jpg' with label 'foo.txt'.

Composed by Ziv Attias.
"""

import os
import random
import shutil

# The base path to dataset
BASE_PATH = "../datasets/coin_detector/"


def sanitize_filenames(directory):
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        new_filename = filename.lower()
        new_path = os.path.join(directory, new_filename)
        if old_path != new_path:
            os.rename(old_path, new_path)


def split_data():
    # Preparing images and labels directory paths
    images_dir = os.path.join(os.path.dirname(__file__), BASE_PATH, "images")
    labels_dir = os.path.join(os.path.dirname(__file__), BASE_PATH, "labels")

    # Validating directories existance
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(
            f"One or both of the directories '{images_dir}' and '{labels_dir}' do not exist. Aborting."
        )
        return

    # Sanitize filenames in images and labels directories
    sanitize_filenames(images_dir)
    sanitize_filenames(labels_dir)

    # Ensuring train, validation, and test directories exist within images and labels directories
    for subdir in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, subdir), exist_ok=True)

    # Getting all image files and shuffling the array
    # (!) You can comment out random.shuffle & random.seed if you want to compare models and performance with others, who also must not shuffle
    all_files = [
        f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))
    ]
    random.seed(42)
    random.shuffle(all_files)

    # Calculating the split indices
    # (!) You can change the split ratios
    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.2

    total_files = len(all_files)
    train_count = int(TRAIN_RATIO * total_files)
    val_count = int(VALIDATION_RATIO * total_files)

    train_files = all_files[:train_count]
    val_files = all_files[train_count : train_count + val_count]
    test_files = all_files[train_count + val_count :]

    def move_files(file_list: list[str], target_dir: str):
        for file in file_list:
            image_file = os.path.join(images_dir, file)
            label_file = os.path.join(labels_dir, file.replace(".jpg", ".txt"))

            if os.path.exists(image_file):
                shutil.move(image_file, os.path.join(images_dir, target_dir))
            if os.path.exists(label_file):
                shutil.move(label_file, os.path.join(labels_dir, target_dir))

    move_files(train_files, "train")
    print(f"Moved {len(train_files)} files to train")

    move_files(val_files, "val")
    print(f"Moved {len(val_files)} files to val")

    move_files(test_files, "test")
    print(f"Moved {len(test_files)} files to test")


if __name__ == "__main__":
    split_data()
