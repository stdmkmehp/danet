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

class LCDSegmentation(BaseDataset):
    NUM_CLASS = 19
    def __init__(self, root='../datasets', split='city',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(LCDSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)

        self.images, self.img_folder = _get_lcd_image(root, split)
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
            raise RuntimeError("In LCDSegmentation. Mode must be vis(visualizing)!")
        
    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_lcd_image(folder, split='city'):
    def get_image(img_folder, img_suffix):
        # assert exists
        assert os.path.exists(img_folder), "Please download the dataset!!"
        img_paths = []
        for filename in os.listdir(img_folder):
            if filename.endswith(img_suffix):
                imgpath = os.path.join(img_folder, filename)
                img_paths.append(imgpath)
        return img_paths
    if split not in {'city', 'college', 'euroc', 'kitti00', 'kitti05', 'malaga6l'}:
        raise RuntimeError("Dataset split must be one of 'city', 'college', 'euroc', 'kitti00', 'kitti05' and 'malaga6l'.")
    elif split in {'city', 'college', 'malaga6l'}:
        img_folder = os.path.join(folder, split)
        img_paths = get_image(img_folder, ".jpg")
    else:
        img_folder = os.path.join(folder, split)
        img_paths = get_image(img_folder, ".png")
    return img_paths, img_folder