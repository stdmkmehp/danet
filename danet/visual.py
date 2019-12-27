import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

IMG_Path = "/home/lab404/zw/DANet/datasets/kitti05"
IMG_Folder = "/home/lab404/zw/DANet/danet/outdir/kitti05"

imNames = sorted(os.listdir(IMG_Path))

plt.ion() # interactive mode
fig = plt.figure()
figOrig = fig.add_subplot(221)
figColor = fig.add_subplot(222)
figGray = fig.add_subplot(223)
figMerge = fig.add_subplot(224)
figOrig.set_title('origin')
figColor.set_title('color')
figGray.set_title('gray')
figMerge.set_title('merge')
'''
imgOrig  = [Image.open(os.path.join(IMG_Path, name)).convert('RGB') for name in imNames]
imgColor = [Image.open(os.path.join(IMG_Folder, 'color', name)).convert('RGB') for name in imNames]
imgGray  = [Image.open(os.path.join(IMG_Folder, 'gray', name)).convert('RGB') for name in imNames]
imgMerge = [Image.open(os.path.join(IMG_Folder, 'merge', name)).convert('RGB') for name in imNames]
for i in range(len(imNames)):
    figOrig.imshow(imgOrig[i])
    figColor.imshow(imgColor[i])
    figGray.imshow(imgGray[i])
    figMerge.imshow(imgMerge[i])
    plt.suptitle(imNames[i])
    plt.show()
    plt.pause(0.1)
'''
for name in imNames:
    figOrig.imshow(Image.open(os.path.join(IMG_Path, name)).convert('RGB'))
    figColor.imshow(Image.open(os.path.join(IMG_Folder, 'color', name)).convert('RGB'))
    figGray.imshow(Image.open(os.path.join(IMG_Folder, 'gray', name)).convert('RGB'))
    figMerge.imshow(Image.open(os.path.join(IMG_Folder, 'merge', name)).convert('RGB'))
    plt.suptitle(name)
    plt.show()
    plt.pause(0.5)