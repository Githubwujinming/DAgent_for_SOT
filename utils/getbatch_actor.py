import torch
import numpy as np
from torch.autograd import Variable
from utils.crop_image import crop_image_actor


def getbatch_actor(img, boxes):
    crop_size = 107

    num_boxes = boxes.shape[0]
    imo_g = np.zeros([num_boxes, crop_size, crop_size, 3])
    imo_l = np.zeros([num_boxes, crop_size, crop_size, 3])

    for i in range(num_boxes):
        bbox = boxes[i]
        img_crop_l, img_crop_g, out_flag = crop_image_actor(img, bbox)

        imo_g[i] = img_crop_g
        imo_l[i] = img_crop_l

    imo_g = imo_g.transpose(0, 3, 1, 2).astype('float32')
    imo_g = (imo_g - 128.)/255.
    imo_g = torch.from_numpy(imo_g)
    imo_g = Variable(imo_g)
    imo_g = imo_g.cuda()
    imo_l = imo_l.transpose(0, 3, 1, 2).astype('float32')
    imo_l = (imo_l - 128.)/255.
    imo_l = torch.from_numpy(imo_l)
    imo_l = Variable(imo_l)
    imo_l = imo_l.cuda()

    return imo_g, imo_l, out_flag