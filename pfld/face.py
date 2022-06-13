from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
import pickle
import random
import torchvision.transforms.functional as TF



class FaceDataset(Dataset):
    def __init__(self, prefix, transform, coordinate_transform=None, do_train=True) -> None:
        super().__init__()
        self.prefix = prefix
        self.transform = transform
        self.coordinate_transform = coordinate_transform
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
            if self.coordinate_transform is not None:
                return self.coordinate_transform((img, norm_landmarks))
            else:
                return img, norm_landmarks.reshape(136)
        else:
            return self.imgs[index], img

class RandomFlip(object): 
    def __init__(self, probability=0.5):
        assert 0 <= probability <= 1
        self.prob = probability 

    def __call__(self, sample):
        image, landmarks = sample
        if self.prob > random.random():
            lm = landmarks.copy()
            lm[:, 0] = 1 - lm[:, 0]
            return TF.hflip(image), lm
        else:
            return image, landmarks

class RandomRotate(object): 
    def __init__(self, degree=0):
        assert degree >= 0
        self.degree = degree

    def __call__(self, sample):
        image, landmarks = sample
        c, h, w = image.shape
        deg = random.uniform(-self.degree, self.degree)
        rad = np.deg2rad(deg)
        rotation_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad),  np.cos(rad)]
        ])
        trans_cor = ((landmarks - 0.5) @ rotation_matrix.T) + 0.5
        if np.any(trans_cor < 0) or np.any(trans_cor > 1):
            return image, landmarks
        else:
            return TF.rotate(image, -deg), trans_cor

class RandomMask(object): 
    def __init__(self, ratio=0):
        assert 0 <= ratio <= 1
        self.ratio = ratio

    def __call__(self, image):
        c, h, w = image.shape
        mask_left = random.randrange(w - 1)
        mask_top = random.randrange(h - 1)
        mask_h = int(random.uniform(0, self.ratio) * h)
        mask_w = int(random.uniform(0, self.ratio) * w)
        if mask_h == 0 or mask_w == 0:
            return image
        mask_right = min(w - 1, mask_left + mask_w)
        mask_bottom = min(h - 1, mask_top + mask_h)
        img = image.clone()
        img[:, mask_top:mask_bottom, mask_left:mask_right] = torch.zeros(c, mask_bottom - mask_top, mask_right - mask_left)
        return img