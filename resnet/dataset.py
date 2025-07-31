from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from prepare_data import load_cropped_images, load_and_augment_labels
import config


class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for handling image-label pairs.

    Args:
        images (numpy.ndarray or torch.Tensor): Array of image data.
        labels (numpy.ndarray or torch.Tensor): Corresponding labels for the images.
    """
    def __init__(self, images, labels):
        self.images = images  # Store image data
        self.labels = labels  # Store corresponding labels

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, label) at the given index.
        """
        return self.images[idx], self.labels[idx]


def get_dataloaders():
    """
    Loads data and returns PyTorch DataLoader objects for training and testing.

    Returns:
        tuple: (train_loader, test_loader) where each is a PyTorch DataLoader.
    """
    images = load_cropped_images()  # Load preprocessed image data
    labels, _, _ = load_and_augment_labels()  # Load and augment labels; ignore mean and std

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=config.test_size, random_state=42  # Split data into training and testing sets
    )

    train_dataset = CustomDataset(train_images, train_labels)  # Wrap training data in a PyTorch Dataset
    test_dataset = CustomDataset(test_images, test_labels)  # Wrap testing data in a PyTorch Dataset

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)  # Create training DataLoader with shuffling
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)  # Create testing DataLoader without shuffling

    return train_loader, test_loader  # Return both DataLoaders


def get_raw_split():
    """
    Loads and splits raw image and label data, along with normalization statistics.

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels, mean, std)
    """
    images = load_cropped_images()  # Load preprocessed image data
    labels, mean, std = load_and_augment_labels()  # Load and augment labels along with mean and std

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=config.test_size, random_state=42  # Split into training and testing sets
    )

    return train_images, train_labels, test_images, test_labels, mean, std  # Return raw splits and normalization stats
