from io import BytesIO
import numpy as np
import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, img_path, label_path, transform, resolution=8):
        self.env_img = lmdb.open(
            img_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.env_label = lmdb.open(
            label_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env_img:
            raise IOError('Cannot open lmdb dataset', img_path)
        if not self.env_label:
            raise IOError('Cannot open lmdb dataset', label_path)

        with self.env_img.begin(write=False) as txn_img:
            self.img_length = int(txn_img.get('length'.encode('utf-8')).decode('utf-8'))

        with self.env_label.begin(write=False) as txn_label:
            self.label_length = int(txn_label.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        if self.img_length == self.label_length:
            return self.img_length
        else:
            raise Exception("Length of image dataset and label dataset do not match")

    def __getitem__(self, index):
        with self.env_img.begin(write=False) as txn_img:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn_img.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        with self.env_label.begin(write=False) as txn_label:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            label = np.float32(txn_label.get(key))

        return img, label
