import torch
import timm  
import torch.nn as nn
import torch.optim as optim
from data_loader import dataset_loader
import tqdm
# Configuration
model_name = "mobilevit_xxs" ## TRY: mobilevit_xxs, mobilevit_xs , mobilevit_s
num_epochs = 10
learning_rate = 0.001
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                      "cpu")
PATH_TO_TRAIN_DATA = "../../Dataset-Split/train"
PATH_TO_VAL_DATA = "../../Dataset-Split/test"
BATCH_SIZE = 32 
NUMBER_WORKERS = 4

dataset_loader = dataset_loader(PATH_TO_TRAIN_DATA, PATH_TO_VAL_DATA, BATCH_SIZE, NUMBER_WORKERS)


# Load pre-trained MobileViT
model = timm.create_model(model_name, pretrained=True, num_classes=3)
model = model.to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, train_loader, val_loader, optimizer, loss_fn, epochs=10):
    for epoch in tqdm.tqdm(range(epochs),desc="Epochs"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm.tqdm(train_loader, colour="green", desc=f"Images {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), f"{model_name}_no_augmentation_finetuned.pt")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


if __name__ == "__main__":
    train(model, dataset_loader.train_loader, dataset_loader.val_loader, optimizer, loss_fn, epochs=num_epochs)