import os
import numpy as np
from PIL import Image
import shutil


def load_image(image_path, target_size=(224, 224)):
    """
    Loads an image, converts it to RGB, resizes it to the target size,
    and normalizes the pixel values to the range [0, 1].

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (width, height).

    Returns:
        np.ndarray: Preprocessed image array, or None if loading fails.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img).astype("float32") / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def extract_ground_truth(filename):
    """
    Extracts the ground truth label from the image filename.
    Assumes the filename starts with the true label followed by an underscore or dash.
    For example, 'inscriptions_001.jpg' returns 'inscriptions'.

    Args:
        filename (str): Filename of the image.

    Returns:
        str: Extracted label.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    # Try underscore as a separator
    if "_" in name:
        label = name.split('_')[0]
    elif "-" in name:
        label = name.split('-')[0]
    else:
        label = name  # Fall back to the full name if no separator found
    return label


def list_images(directory, extensions=('jpg', 'jpeg', 'png')):
    """
    Lists all image files in a directory with the given extensions.

    Args:
        directory (str): Path to the directory.
        extensions (tuple): Tuple of acceptable file extensions.

    Returns:
        list: List of file paths matching the extensions.
    """
    files = []
    for file in os.listdir(directory):
        if file.lower().endswith(extensions):
            files.append(os.path.join(directory, file))
    return files


def copy_file(source, destination):
    """
    Copies a file from source to destination.

    Args:
        source (str): Path to the source file.
        destination (str): Path to the destination.
    """
    try:
        shutil.copy2(source, destination)
    except Exception as e:
        print(f"Error copying {source} to {destination}: {e}")
