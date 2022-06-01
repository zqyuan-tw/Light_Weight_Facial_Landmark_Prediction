from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pickle

class FaceDataset(Dataset):
    def __init__(self, prefix, transform, do_train=True) -> None:
        super().__init__()
        self.prefix = prefix
        self.transform = transform
        self.landmarks = None
        if do_train:
            with open(os.path.join(self.prefix, 'annot.pkl'), 'rb') as f:
                self.imgs, self.landmarks = pickle.load(f)
            self.landmarks = np.array(self.landmarks)
        else:
            self.imgs = os.listdir(self.prefix)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.transform(Image.open(os.path.join(self.prefix, self.imgs[index])))
        C, H, W = img.shape
        if self.landmarks is not None:
            norm_landmarks = np.stack((self.landmarks[index, :, 0] / W, self.landmarks[index, :, 1] / H), axis=1)
            return img, norm_landmarks
        else:
            return img, 0