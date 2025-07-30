import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
from prepare_data import load_images

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_dataloaders():
    images, labels = load_images()

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=config.test_size, random_state=42
    )

    train_dataset = CustomDataset(train_images, train_labels)
    test_dataset = CustomDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

    return train_loader, test_loader

