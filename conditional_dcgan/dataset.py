import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
from prepare_data import load_images

class CustomDataset(Dataset):
    """
    PyTorch Dataset for pairing microstructure images with their corresponding labels.
    
    Args:
        images (Tensor): Tensor of image data with shape [N, C, H, W]
        labels (Tensor): Corresponding label tensor of shape [N]
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images)

    def __getitem__(self, idx):
        """Fetches the image and label at a given index."""
        return self.images[idx], self.labels[idx]


def get_dataloaders(crop_images):
    """
    Loads image data and splits it into training and testing sets, 
    returning PyTorch DataLoaders for each.
    
    Args:
        crop_images (str): Directory path containing cropped images.
    
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # Load all cropped images and labels
    images, labels = load_images(crop_images)

    # Split dataset into training and testing subsets
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=config.test_size, random_state=42
    )

    # Wrap in PyTorch Dataset class
    train_dataset = CustomDataset(train_images, train_labels)
    test_dataset = CustomDataset(test_images, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

    return train_loader, test_loader
