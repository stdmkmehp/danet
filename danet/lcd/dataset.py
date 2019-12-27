# =====================================================================
# dataset.py - CNNs for loop-closure detection in vSLAM systems.
# Copyright (C) 2018  Zach Carmichael
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =====================================================================
from scipy.io import loadmat
from scipy.ndimage import imread
import numpy as np

import os
import requests
import zipfile
import sys
from glob import glob

# === DATASET VARS ===
LCD_IMG_PATH = {
    'euroc'     : "/home/lab404/zw/DANet/datasets/euroc",
    'kitti00'   : "/home/lab404/zw/DANet/datasets/kitti00",
    'kitti05'   : "/home/lab404/zw/DANet/danet/outdir/kitti05/merge",
    'malaga6l'  : "/home/lab404/zw/DANet/datasets/malaga6l",
    'city'      : "/home/lab404/zw/DANet/datasets/city",
    'college'   : "/home/lab404/zw/DANet/datasets/college"
}
LCD_GT_PATH = {
    'euroc'     : "/home/lab404/zw/datasets/lcdGroundTruth/EuRoCMH05GroundTruth.mat",
    'kitti00'   : "/home/lab404/zw/datasets/lcdGroundTruth/KITTI00GroundTruth.mat",
    'kitti05'   : "/home/lab404/zw/datasets/lcdGroundTruth/KITTI05GroundTruth.mat",
    'malaga6l'  : "/home/lab404/zw/datasets/lcdGroundTruth/Malaga6LGroundTruth.mat",
    'city'      : "/home/lab404/zw/datasets/lcdGroundTruth/CityCentreGroundTruth.mat",
    'college'   : "/home/lab404/zw/datasets/lcdGroundTruth/NewCollegeGroundTruth.mat"
}

def download_file(url, file_name):
    """Downloads a file to destination

    Code adapted from:
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url:       URL of file to download
        file_name: Where to write downloaded file
    """
    # Ensure destination exists
    dest_dir = os.path.dirname(file_name)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    with open(file_name, 'wb') as f:
        print('Downloading {} from {}'.format(file_name, url))
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                # Output progress
                complete = dl / total_length
                done = int(50 * complete)
                sys.stdout.write('\r[{}{}] {:6.2f}%'.format('=' * done, ' ' * (50 - done),
                                                            complete * 100))
                sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

def get_dataset_from_path(IMG_PATH, GT_PATH, img_suffix, assetlen, debug):
    debug_amt = 25
    print('Loading the dataset from {}...'.format(IMG_PATH))
    # Load images
    print('Loading images')
    '''
    if not os.path.isfile(CITY_IMGZIP_PATH):
        download_file(CITY_IMG_URL, CITY_IMGZIP_PATH)
    if not os.path.isdir(CITY_IMG_PATH):
        # Unzip archive
        print('Unzipping {} to {}'.format(CITY_IMGZIP_PATH, CITY_DATA_DIR))
        with zipfile.ZipFile(CITY_IMGZIP_PATH, 'r') as zip_handle:
            zip_handle.extractall(CITY_DATA_DIR)
    '''
    # Sort by image number
    img_names = sorted(glob(os.path.join(IMG_PATH, img_suffix)))
    assert len(img_names) == assetlen
    if debug:
        print('Using fewer images ({}) per debug flag...'.format(
            debug_amt))
        img_names = img_names[:debug_amt]
    imgs = np.asarray([imread(img, mode='RGB') for img in img_names])
    # Load GT
    if not os.path.isfile(GT_PATH):
        # download_file(CITY_GT_URL, CITY_GT_PATH)
        raise RuntimeError('No ground truth!')
    print('Loading ground truth from {}...'.format(GT_PATH))
    gt = loadmat(GT_PATH)['truth']
    if debug:
        gt = gt[:debug_amt, :debug_amt]
    return imgs, gt

def get_dataset(name, debug=False):
    debug_amt = 25
    if name.lower() == 'euroc':
        return get_dataset_from_path(LCD_IMG_PATH[name.lower()], LCD_GT_PATH[name.lower()],'*.png', 2273, debug)
    elif name.lower() == 'kitti00':
        return get_dataset_from_path(LCD_IMG_PATH[name.lower()], LCD_GT_PATH[name.lower()],'*.png', 4541, debug)
    elif name.lower() == 'kitti05':
        return get_dataset_from_path(LCD_IMG_PATH[name.lower()], LCD_GT_PATH[name.lower()],'*.png', 2761, debug)
    elif name.lower() == 'malaga6l':
        return get_dataset_from_path(LCD_IMG_PATH[name.lower()], LCD_GT_PATH[name.lower()],'*.jpg', 3474, debug)
    elif name.lower() == 'city':  # city centre dataset
        return get_dataset_from_path(LCD_IMG_PATH[name.lower()], LCD_GT_PATH[name.lower()],'*.jpg', 2474, debug)
    elif name.lower() == 'college':  # new college dataset
        return get_dataset_from_path(LCD_IMG_PATH[name.lower()], LCD_GT_PATH[name.lower()],'*.jpg', 2146, debug)
    elif name.lower() == 'tsukuba':  # new tsukuba dataset
        raise NotImplementedError
    else:
        raise ValueError('lcd_dataset must be one of {}.'.format(LCD_IMG_PATH.keys()))
    return imgs, gt
