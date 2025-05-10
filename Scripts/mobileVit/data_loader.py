from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

class dataset_loader():
    def __init__(self, path_to_train_data, path_to_val_data, batch_size=32,numbr_workers=4):
        # Define data transformations 
        # self.train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),  # resize to 224x224
        # transforms.RandomHorizontalFlip(),  # flip horizontally
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # change colors and brightness
        # transforms.ToTensor(),  # convert to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # without augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
        transforms.Resize(256), # resize to 256x256
        transforms.CenterCrop(224),  # crop to 224x224
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
          ])
        # Load datasets
        train_dataset = ImageFolder(root=path_to_train_data, transform=self.train_transform)
        val_dataset = ImageFolder(root=path_to_val_data, transform=self.val_transform)

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=numbr_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=numbr_workers)