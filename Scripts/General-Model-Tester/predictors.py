import os
import numpy as np
from PIL import Image

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

# PyTorch imports
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms


class TensorFlowPredictor:
    def __init__(self, model_type, weights_path, target_size=(224, 224)):
        """
        Initializes a TensorFlow-based predictor.

        Args:
            model_type (str): Type of the model (e.g., "efficientnetb0").
            weights_path (str): Path to the saved weights file.
            target_size (tuple): Desired image size for input.
        """
        self.model_type = model_type.lower()
        self.target_size = target_size

        if self.model_type == "efficientnetb0":
            base_model = EfficientNetB0(weights="imagenet", include_top=False,
                                        input_shape=(target_size[0], target_size[1], 3))
            self.model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(3, activation="softmax")  # Assuming 3 classes
            ])
        else:
            raise ValueError("Unsupported TensorFlow model type.")

        # Load weights
        self.model.load_weights(weights_path)
        print("✅ TensorFlow model loaded successfully!")

    def predict(self, image_path):
        """
        Loads and preprocesses an image from the given path, then returns the predicted class index.

        Args:
            image_path (str): Path to the image file.

        Returns:
            int or None: Predicted class index, or None if processing fails.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.target_size)
            img_array = np.array(img).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction)
        return predicted_class


class PyTorchPredictor:
    def __init__(self, model_type, weights_path, target_size=(224, 224), device=None):
        """
        Initializes a PyTorch-based predictor.

        Args:
            model_type (str): Type of the model (e.g., "mobilevit_s", "mobilevit_xs", "mobilevit_xxs").
            weights_path (str): Path to the saved checkpoint.
            target_size (tuple): Desired image size for input.
            device (torch.device, optional): Device to run the model on.
        """
        self.model_type = model_type.lower()
        self.target_size = target_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type in ["mobilevit_s", "mobilevit_xs", "mobilevit_xxs"]:
            self.model = timm.create_model(model_type, pretrained=True, num_classes=3)
        else:
            raise ValueError("Unsupported PyTorch model type.")

        # Load the state dictionary
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations: resize, to tensor, and normalization
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("✅ PyTorch model loaded successfully!")

    def predict(self, image_path):
        """
        Loads and preprocesses an image from the given path, then returns the predicted class index.

        Args:
            image_path (str): Path to the image file.

        Returns:
            int or None: Predicted class index, or None if processing fails.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        img = self.transform(img)
        img = img.unsqueeze(0)  # Add batch dimension
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            predicted = torch.argmax(outputs, dim=1)
        return predicted.item()
