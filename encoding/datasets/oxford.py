###########################################################################
# Created by: zw
# Email: xl2013919@sina.cn
# Copyright (c) 2019
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from tqdm import tqdm
from .base import BaseDataset

class OxfordSegmentation(BaseDataset):
    BASE_DIR = 'oxford'
    NUM_CLASS = 19
    def __init__(self, root='../datasets', split='citycentre',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(OxfordSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images = _get_oxford_image(root, split)
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        else:
            raise RuntimeError("In OxfordSegmentation. Mode must be vis(visualizing)!")
        
    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_oxford_image(folder, split='citycentre'):
    def get_image(img_folder):
        img_paths = []
        for filename in os.listdir(img_folder):
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                img_paths.append(imgpath)
        return img_paths
    if split == 'citycentre':
        img_folder = os.path.join(folder, 'CityCentre/Images')
        img_paths = get_image(img_folder)
    elif split == 'newcollege':
        img_folder = os.path.join(folder, 'NewCollege/Images')
        img_paths = get_image(img_folder)
    else:
        raise RuntimeError("Dataset split must be one of 'citycentre' and 'newcollege'.")
    return img_paths
