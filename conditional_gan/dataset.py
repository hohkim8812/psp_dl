import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
from prepare_data import load_images

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset to handle image-label pairs.

    Args:
        images (torch.Tensor): Input image data.
        labels (torch.Tensor): Corresponding labels.

    Returns:
        Tuple of (image, label) when indexed.
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple: (image, label) at given index.
        """
        return self.images[idx], self.labels[idx]


def get_dataloaders(crop_images):
    """
    Loads data, splits into training and testing sets, wraps them in DataLoaders.

    Returns:
        tuple: (train_loader, test_loader)
            - train_loader (DataLoader): DataLoader for training data.
            - test_loader (DataLoader): DataLoader for testing data.
    """
    # Load full dataset (images and labels)
    images, labels = load_images(crop_images)

    # Split into train/test subsets with fixed random seed for reproducibility
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=config.test_size,
        random_state=42
    )

    # Wrap splits into Dataset objects
    train_dataset = CustomDataset(train_images, train_labels)
    test_dataset = CustomDataset(test_images, test_labels)

    # Wrap Dataset into DataLoader for batch processing
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader
