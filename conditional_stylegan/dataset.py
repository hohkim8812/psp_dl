from io import BytesIO
import numpy as np
import lmdb
from PIL import Image
from torch.utils.data import Dataset

class MultiResolutionDataset(Dataset):
    def __init__(self, img_path, label_path, transform, resolution=8):
        """
        Dataset class to read images and corresponding labels from LMDB databases.

        Args:
            img_path (str): Path to the LMDB database containing images.
            label_path (str): Path to the LMDB database containing labels.
            transform (callable): Image transformation function (e.g., preprocessing, augmentation).
            resolution (int): Resolution level of the images to retrieve (e.g., 8, 16, ..., 128).
        """
        # Open LMDB environment for images
        self.env_img = lmdb.open(
            img_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Open LMDB environment for labels
        self.env_label = lmdb.open(
            label_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Ensure both environments are accessible
        if not self.env_img:
            raise IOError('Cannot open lmdb dataset', img_path)
        if not self.env_label:
            raise IOError('Cannot open lmdb dataset', label_path)

        # Load dataset lengths
        with self.env_img.begin(write=False) as txn_img:
            self.img_length = int(txn_img.get('length'.encode('utf-8')).decode('utf-8'))

        with self.env_label.begin(write=False) as txn_label:
            self.label_length = int(txn_label.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.
        Raises an exception if image and label counts do not match.
        """
        if self.img_length == self.label_length:
            return self.img_length
        else:
            raise Exception("Length of image dataset and label dataset do not match")

    def __getitem__(self, index):
        """
        Retrieve image and corresponding label for a given index.

        Args:
            index (int): Sample index

        Returns:
            img (Tensor): Transformed image
            label (float32): Associated label
        """
        # Load image from LMDB
        with self.env_img.begin(write=False) as txn_img:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn_img.get(key)

        # Convert image bytes to PIL image and apply transform
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        # Load label from LMDB
        with self.env_label.begin(write=False) as txn_label:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            label = np.float32(txn_label.get(key))

        return img, label
