from PIL import Image
import numpy as np

citypallete = { 0:(0,255,0), 1:(0,255,0), 2:(0,255,0), 3:(0,255,0), 4:(0,255,0), 5:(0,255,0), 6:(255,0,0),
                7:(0,255,0), 8:(0,0,255), 9:(0,0,255), 10:(0,0,255), 11:(255,0,0), 12:(255,0,0),
                13:(255,0,0), 14:(255,0,0), 15:(255,0,0), 16:(255,0,0), 17:(255,0,0), 18:(255,0,0) }

def get_mask_pure(npmask, imgpath, dataset='cityscapes'):
    # recovery boundary
    if dataset == 'pascal_voc':
        npmask[npmask==21] = 255
    if dataset != 'cityscapes':
        raise RuntimeError("Wrong in get_mask_pure(). dataset mnust be cityscapes!")
    # put colormap
    img = Image.open(imgpath)
    imgdata = img.load()
    width, height = img.size
    _, rows, cols = npmask.shape
    assert rows == height and cols == width
    for i in range(rows):
        for j in range(cols):
            if npmask[0, i, j] in citypallete:
                imgdata[j,i] = citypallete[npmask[0, i, j]]
            else:
                raise RuntimeError("Some pixels are beyond 19 categories.")
                imgdata[j,i] = (255,255,255)
    return img, Image.fromarray(npmask.squeeze().astype('uint8'))