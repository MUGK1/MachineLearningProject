import streamlit as st
from PIL import Image
from utils.inference import load_model_and_predictor
import tempfile
import os



# === Page config ===
st.set_page_config(
    page_title="Arabic Document Classifier Demo",
    layout="centered",
    page_icon="📜"
)

st.title("📜 Historical Arabic Document Classifier")
st.markdown("Upload an image and select a model to classify it!")

# === Sidebar ===
st.sidebar.title("⚙️ Settings")

# 1. Model Selection
model_options = [
    "EfficientNetB0",
    "MobileViT S",
    "MobileViT XS",
    "MobileViT XXS",
]
selected_model = st.sidebar.selectbox("Select Model", model_options)


# 2. Augmentation Options
augmentation_options = [
    "Augmented",
    "No Augmentation"
]

selected_augmentation = st.sidebar.selectbox("Select Augmentation", augmentation_options)


# 3. Image Uploader
uploaded_file = st.sidebar.file_uploader("Upload Historical Image", type=["png", "jpg", "jpeg"])

# 3. Classify Button
classify_button = st.sidebar.button("🧠 Classify")



# === Main Area ===
if uploaded_file:
    # Show image preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Historical Document", use_container_width=True)

    if classify_button:
        with st.spinner("⏳ Loading model..."):

            # === Map model selection to file path ===
            model_path_map = {
                "EfficientNetB0 (Augmented)": "models/efficientnetb0.weights.h5",
                "EfficientNetB0 (No Augmentation)": "models/efficientnetb0_no_augmentation.weights.h5",
                "MobileViT S (Augmented)": "models/mobilevit_s_finetuned.pt",
                "MobileViT S (No Augmentation)": "models/mobilevit_s_no_augmentation_finetuned.pt",
                "MobileViT XS (Augmented)": "models/mobilevit_xs_finetuned.pt",
                "MobileViT XS (No Augmentation)": "models/mobilevit_xs_no_augmentation_finetuned.pt",
                "MobileViT XXS (Augmented)": "models/mobilevit_xxs_finetuned.pt",
                "MobileViT XXS (No Augmentation)": "models/mobilevit_xxs_no_augmentation_finetuned.pt",
            }

            model_name = selected_model.lower().replace(" ", "_").replace("(", "").replace(")", "")
            model_path = model_path_map[f"{selected_model} ({selected_augmentation})"]

            predictor, framework = load_model_and_predictor(model_name, model_path)

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                image.save(temp_file.name)
                temp_path = temp_file.name

            pred_idx = predictor.predict(temp_path)
            os.unlink(temp_path)  # Delete temp file

            labels = ["Inscriptions", "Manuscripts", "Other"]
            prediction = labels[pred_idx]

            st.success(f"✅ Prediction: **{prediction}** (Class {pred_idx})")

else:
    st.info("📎 Please upload an image to get started.")

