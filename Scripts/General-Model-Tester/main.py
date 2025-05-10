import os
from tqdm import tqdm
from utils import list_images, copy_file, extract_ground_truth
from predictors import TensorFlowPredictor, PyTorchPredictor
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def main():
    # Welcome and explanation messages
    print("👋 Welcome to the Generic Model Tester for Historical Manuscript Classification!")
    print("🔍 This script will:")
    print("   ✅ Load your specified model weights/checkpoint")
    print("   ✅ Process each image in your input folder")
    print("   ✅ Classify images and copy them to designated output folders")
    print("   ✅ Compute evaluation metrics (accuracy, precision, recall, and F1 score)")
    print("💡 Note: Make sure your image filenames start with the true class label (e.g., inscriptions_001.jpg)")
    print("--------------------------------------------------------------------------------\n")

    # Ask user for the framework
    while True:
        framework = input("Which framework do you want to use? (tensorflow/pytorch): ").strip().lower()
        if framework in ["tensorflow", "pytorch"]:
            break
        else:
            print("❌ Invalid input. Please enter either 'tensorflow' or 'pytorch'.")


    # Ask user for the model type
    model_type = None
    # ask if they want to use model trained on augmented data or no
    augmentedData = input("Do you want to use the model trained on augmented data? (y/n): ").strip().lower()
    if framework == "tensorflow":
        model_type = "efficientnetb0"
    elif framework == "pytorch":
        model_type = input(
        "Enter the model type (e.g., 'MV_s, MV_xs, or MV_xxs' for PyTorch): ").strip().lower()

    # map MV_s, MV_xs, MV_xxs to mobilevit_s, mobilevit_xs, mobilevit_xxs
    if model_type == "mv_s":
        model_type = "mobilevit_s"
    elif model_type == "mv_xs":
        model_type = "mobilevit_xs"
    elif model_type == "mv_xxs":
        model_type = "mobilevit_xxs"

    print(model_type)


    # Ask for the weights/checkpoint path based on the chosen framework
    if framework == "tensorflow":
        if augmentedData == "y":
            weights_path = '../efficientnetb0.weights.h5'
        else:
            weights_path = '../efficientnetb0_no_augmentation.weights.h5'
    else:
        if model_type == "mobilevit_s":
            if augmentedData == "y":
                weights_path = '../mobileVit/mobilevit_s_finetuned.pt'
            else:
                weights_path = '../mobileVit/mobilevit_s_no_augmentation_finetuned.pt'
        elif model_type == "mobilevit_xs":
            if augmentedData == "y":
                weights_path = '../mobileVit/mobilevit_xs_finetuned.pt'
            else:
                weights_path = '../mobileVit/mobilevit_xs_no_augmentation_finetuned.pt'
        elif model_type == "mobilevit_xxs":
            if augmentedData == "y":
                weights_path = '../mobileVit/mobilevit_xxs_finetuned.pt'
            else:
                weights_path = '../mobileVit/mobilevit_xxs_no_augmentation_finetuned.pt'


    # Ask for input and output folder paths
    # input_folder = input("Enter the path to the folder containing images to test: ").strip()
    outputFolderName = input("Enter the Output Folder Name ONLY! (e.g. Classified-Test-1): ").strip()

    input_folder = "../../ValidationDataUpdated"
    output_folder = f"../../{outputFolderName}"

    # Summary of provided parameters
    print("\n🚀 Starting with the following parameters:")
    print(f"   Framework: {framework}")
    print(f"   Model Type: {model_type}")
    print(f"   Weights/Checkpoint Path: {weights_path}")
    print(f"   Input Folder: {input_folder}")
    print(f"   Output Folder: {output_folder}")
    print("--------------------------------------------------------------------------------\n")

    # Ask user to confirm before proceeding
    proceed = input("Are you ready to proceed? (y/n): ")
    if proceed.strip().lower() != "y":
        print("❌ Operation cancelled by user. Exiting... 👋")
        return

    # Define the mapping from predicted class indices to class labels
    class_labels = {0: "inscriptions", 1: "manuscripts", 2: "other"}

    # Instantiate the appropriate predictor
    print("\n🛠️ Loading the model. Please wait...")
    if framework == "tensorflow":
        predictor = TensorFlowPredictor(model_type, weights_path)
    else:
        predictor = PyTorchPredictor(model_type, weights_path)

    # Prepare output subdirectories for each class label
    print("📁 Setting up output directories...")
    for label in class_labels.values():
        os.makedirs(os.path.join(output_folder, label), exist_ok=True)

    ground_truths = []
    predictions = []

    # Get list of image files from the input folder
    image_files = list_images(input_folder)
    print(f"📸 Found {len(image_files)} images in {input_folder}.\n")

    # Process each image
    for image_path in tqdm(image_files, desc="🖼️ Classifying Images"):
        # Extract the ground truth label from the filename
        gt_label = extract_ground_truth(os.path.basename(image_path)).lower()

        # Use the predictor to classify the image (returns class index)
        pred_index = predictor.predict(image_path)
        if pred_index is None:
            print(f"⚠️ Skipping image {image_path} due to prediction error.")
            continue

        pred_label = class_labels.get(pred_index, "unknown")
        ground_truths.append(gt_label)
        predictions.append(pred_label)

        # Copy the image to the corresponding output folder based on prediction
        dest_path = os.path.join(output_folder, pred_label, os.path.basename(image_path))
        copy_file(image_path, dest_path)

        # Instead of normal print (which breaks tqdm), use tqdm.write()
        tqdm.write(f"✅ '{os.path.basename(image_path)}': GT='{gt_label}', Pred='{pred_label}'")

    # Calculate and display evaluation metrics
    print("\n📊 Evaluation Metrics:")
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, average="macro")
    recall = recall_score(ground_truths, predictions, average="macro")
    f1 = f1_score(ground_truths, predictions, average="macro")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}\n")
    print("📑 Classification Report:")
    print(classification_report(ground_truths, predictions))
    print("🎉 Testing completed successfully!")


if __name__ == "__main__":
    main()
