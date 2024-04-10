# This /datasets directory is adapted from TorchSSL: https://github.com/TorchSSL/TorchSSL

from torchvision import transforms
from torch.utils.data import Dataset
from .augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
import numpy as np
import copy

def keep(x):
    return x

class BasicDataset(Dataset):
    '''
    BasicDataset from TorchSSL: https://github.com/TorchSSL/TorchSSL
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    '''

    def __init__(self,
                 data,
                 targets=None,
                 alg=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 *args, **kwargs):
        '''
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        '''
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb

        self.transform = keep
        self.strong_transform = strong_transform = keep
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        '''
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        '''

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target = self.targets[idx]

        # set augmented images

        img = self.data[idx]
        img_w = self.transform(img)
        return idx, img_w, target

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.targets)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


    def __len__(self):
        return len(self.data)
