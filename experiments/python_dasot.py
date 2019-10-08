import vot
import sys
import cv2
import time
import torch
import numpy as np
import collections

sys.path.insert(0,'../modules')
from actor import *
from option import *
from data_prov import *
from sample_generator import *

from modules.SiameseNet import SiameseNet
from modules.tem_policy_base import T_Policy
from modules.SiamFcTracker import SiamFCTracker
from modules.EmbeddingNet import BaselineEmbeddingNet

from utils.init_video import _init_video
from utils.cal_distance import cal_distance
from utils.crop_image import move_crop_tracking
from utils.region_to_bbox import region_to_bbox
from utils.getbatch_actor import getbatch_actor
from utils.compile_results import _compile_results


T_N = 5

def init_actor(actor, image, gt):
    batch_num = 64
    maxiter = 10
    actor = actor.cuda()
    actor.train()
    init_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
    loss_func= torch.nn.MSELoss()
    _, _, out_flag_first = getbatch_actor(np.array(image), np.array(gt).reshape([1, 4]))
    actor_samples = np.round(gen_samples(SampleGenerator('uniform', (image.shape[1],image.shape[0]), 0.3, 1.5, None),
                                         gt, 640, [0.6, 1], [0.9, 1.1]))
    idx = np.random.permutation(actor_samples.shape[0])
    batch_img_g, batch_img_l, _ = getbatch_actor(np.array(image), actor_samples)
    batch_distance = cal_distance(actor_samples, np.tile(gt, [actor_samples.shape[0], 1]))
    batch_distance = np.array(batch_distance).astype(np.float32)
    while (len(idx) < batch_num * maxiter):
        idx = np.concatenate([idx, np.random.permutation(actor_samples.shape[0])])

    pointer = 0
    # torch_image = loader(image.resize((255,255),Image.ANTIALIAS)).unsqueeze(0).cuda() - 128./255.
    for iter in range(maxiter):
        next = pointer + batch_num
        cur_idx = idx[pointer: next]
        pointer = next
        feat = actor(batch_img_l[cur_idx], batch_img_g[cur_idx])
        loss = loss_func(feat, (torch.FloatTensor(batch_distance[cur_idx])).cuda())
        del feat
        actor.zero_grad()
        loss.backward()
        init_optimizer.step()
        if loss.item() < 0.0001:
            deta_flag = 0
            return deta_flag
        deta_flag = 1
    return deta_flag, out_flag_first

class DATracker(object):

    def __init__(self, image, region):


        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)


        self.siam = SiameseNet(BaselineEmbeddingNet())
        # weights_init(siam)
        siamfc_path = "../models/siamfc_pretrained.pth"

        pretrained_siam = torch.load(siamfc_path)
        siam_dict = self.siam.state_dict()
        pretrained_siam = {k: v for k, v in pretrained_siam.items() if k in siam_dict}
        siam_dict.update(pretrained_siam)
        self.siam.load_state_dict(siam_dict)

        self.pi = T_Policy(T_N)

        pretrained_pi_dict = torch.load('../models/template_policy/95600_template_policy.pth')
        pi_dict = self.pi.state_dict()
        pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict}
        # pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict and k.startswith("conv")}
        pi_dict.update(pretrained_pi_dict)
        self.pi.load_state_dict(pi_dict)

        self.actor = Actor()  # .load_state_dict(torch.load("../Models/500_actor.pth"))
        pretrained_act_dict = torch.load("../models/Double_agent/95600_DA_actor.pth")
        actor_dict = self.actor.state_dict()
        pretrained_act_dict = {k: v for k, v in pretrained_act_dict.items() if k in actor_dict}
        actor_dict.update(pretrained_act_dict)
        self.actor.load_state_dict(actor_dict)
        self.tracker = SiamFCTracker(model_path=siamfc_path, gpu_id=0)
        if torch.cuda.is_available():
            self.siam = self.siam.cuda()
            self.pi = self.pi.cuda()
        init_bbox = np.array([left, top, region.width, region. height])
        self.rate = init_bbox[2] / init_bbox[3]
        self.template = self.tracker.init(image, init_bbox)
        self.templates = []
        for i in range(T_N):
            self.templates.append(self.template)
        self.deta_flag, self.out_flag_first = init_actor(self.actor, image, init_bbox)


    def track(self, image):

        np_img = np.array(cv2.resize(image, (255, 255), interpolation=cv2.INTER_AREA)).transpose(2, 0, 1)
        np_imgs = []
        for i in range(T_N):
            np_imgs.append(np_img)

        with torch.no_grad():
            responses = self.siam(torch.Tensor(self.templates).permute(0, 3, 1, 2).float().cuda(), torch.Tensor(np_imgs).float().cuda())
            action = self.pi(responses.permute(1, 0, 2, 3).cuda()).cpu().detach().numpy()
        action_id = np.argmax(action)
        # print(action_id)
        template = self.templates[action_id]
        with torch.no_grad():
            siam_box = tracker.update(image, template)
        siam_box = np.round([siam_box[0], siam_box[1], siam_box[2] -siam_box[0], siam_box[3] - siam_box[1]])
        img_g, img_l, out_flag = getbatch_actor(np.array(image), np.array(siam_box).reshape([1, 4]))
        with torch.no_grad():
            deta_pos = self.actor(img_l, img_g)
        deta_pos = deta_pos.data.clone().cpu().numpy()
        if deta_pos[:, 2] > 0.05 or deta_pos[:, 2] < -0.05:
            deta_pos[:, 2] = 0
        if self.deta_flag or (out_flag and not self.out_flag_first):
            deta_pos[:, 2] = 0

        pos_ = np.round(move_crop_tracking(np.array(siam_box), deta_pos, (image.shape[1], image.shape[0]), self.rate))
        result = pos_
        return vot.Rectangle(result[0], result[1], result[2], result[3])


handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
tracker = DATracker(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    region = tracker.track(image)
    handle.report(region)
