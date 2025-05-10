# Import necessary libraries
import os
import warnings
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping



# Define dataset paths
train_dir = "../Dataset-Split/train"
test_dir = "../Dataset-Split/test"


#----------------------------------------#

# Load and preprocess the dataset

#----------------------------------------#



# Image data generator with normalization
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=40,  # Increased rotation
#     width_shift_range=0.3,  # Increased width shift
#     height_shift_range=0.3,  # Increased height shift
#     shear_range=0.3,  # Increased shearing
#     zoom_range=0.3,  # Random zooming
#     horizontal_flip=True,  # Keep flipping
#     fill_mode="nearest"  # Fill in missing pixels
# )

# without augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalization for training data

test_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalization for test data

# Load training dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images (already resized before, but ensures consistency)
    batch_size=16,  # Number of images processed at once
    class_mode="sparse"  # Multi-class classification
)

# Load testing dataset
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="sparse"
)

# Print class labels mapping
print("Class indices:", train_generator.class_indices)




#----------------------------------------#

# Train the model

#----------------------------------------#

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter("ignore")

# Load EfficientNetB0 model without top layers
# Load EfficientNetB0 model without pretrained weights
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

base_model.trainable = True  # Allow some layers to train

# Define custom classification layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduce feature maps to a single vector
    Dense(128, activation="relu"),  # Fully connected layer
    Dropout(0.5),  # Dropout to reduce overfitting
    Dense(3, activation="softmax")  # 3 classes: inscriptions, manuscripts, other
])

# Compile the model using the legacy optimizer
model.compile(optimizer=RMSprop(learning_rate=0.0001),  # Slightly higher LR
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)


# Train the model
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30,  # Try more epochs, but early stopping will stop if overfitting happens
    callbacks=[reduce_lr, early_stopping]  # Added callbacks
)

# Save model weights instead of full model
model.save_weights("efficientnetb0_notest_augmentation.weights.h5")
print("Training complete! ✅ Model weights saved successfully!")