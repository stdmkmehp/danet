from PIL import Image
import numpy as np

citypallete = { (0,1,2,3,4,5,7) : (0,255,0),
                (6,11,12,13,14,15,16,17,18) : (255,0,0),
                (8,9,10) : (0,0,255) }

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
                imgdata[j,i] = (255,255,255)
            print(imgdata[j,i])
    return img, Image.fromarray(npmask.squeeze().astype('uint8'))
    
# img = Image.open("/home/lab404/zw/DANet/img/zurich_000036_000019_leftImg8bit.png")
# imgdata = img.load()
# width, height = img.size
# for i in range(width):
#     for j in range(height):
#         imgdata[i,j] = (255,255,255)
# img.save('/home/lab404/zw/DANet/img/white.png')
