import os
import torch
import encoding

import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from lcd import lcdutils

from option import Options
args = Options().parse()

# Get the model
if args.model_zoo is not None:
    model = get_model(args.model_zoo, pretrained=True).cuda()
else:
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                    backbone=args.backbone, aux=args.aux,
                                    se_loss=args.se_loss, norm_layer=BatchNorm2d,
                                    base_size=args.base_size, crop_size=args.crop_size,
                                    multi_grid=args.multi_grid, multi_dilation=args.multi_dilation).cuda()
    # resuming checkpoint
    if args.resume_dir is not None:
        args.resume = os.path.join(args.resume_dir, "DANet101.pth.tar")
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
    checkpoint = torch.load(args.resume)
    # strict=False, so that it is compatible with old pytorch saved models
    model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

# Prepare the image
filename = '/home/lab404/zw/DANet/img/zurich_000036_000019_leftImg8bit.png'
img = encoding.utils.load_image(filename).cuda().unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model.evaluate(img)
datasetoffset = 0
predict = torch.max(output, 1)[1].cpu().numpy() + datasetoffset

# Get color pallete for visualization
mask = encoding.utils.get_mask_pallete(predict, 'cityscapes')
mask.save('/home/lab404/zw/DANet/img/output_danet.png')

imgMotion, imgAll = lcdutils.get_mask_pure(predict, filename, 'cityscapes')
imgMotion.save('/home/lab404/zw/DANet/img/output_motion.png')
imgAll.save('/home/lab404/zw/DANet/img/output_all.png')
