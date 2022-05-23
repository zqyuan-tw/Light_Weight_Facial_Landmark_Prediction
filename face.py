from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pickle

class FaceDataset(Dataset):
    def __init__(self, prefix, transform) -> None:
        super().__init__()
        self.prefix = prefix
        self.transform = transform
        with open(os.path.join(dir, 'annot.pkl'), 'rb') as f:
            self.imgs, self.landmarks = pickle.load(f)
        self.landmarks = np.array(self.landmarks)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.transform(Image.open(os.path.join(self.prefix, self.imgs[index])))
        C, H, W = img.shape
        norm_landmarks = np.array([self.landmarks[index, 0] / W, self.landmarks[index, 1] / H])
        return img, norm_landmarks