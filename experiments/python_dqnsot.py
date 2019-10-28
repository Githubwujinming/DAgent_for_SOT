import vot
import sys
import cv2
import time
import torch
import numpy as np
import collections
import os.path

# make sure the paths you need are append
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

from modules.sample_generator import *
from modules.actor import Actor
# from option import *
# from data_prov import *

from modules.QNet_cir import QNet_cir
from modules.SiameseNet import SiameseNet
from modules.tem_policy_base import T_Policy
from modules.SiamFcTracker import SiamFCTracker
from modules.EmbeddingNet import BaselineEmbeddingNet

from utils.np2tensor import np2tensor
from utils.crop_image import crop_image_actor_
from utils.crop_image import move_crop_tracking

project_path = os.path.abspath(os.path.join(basedir, os.path.pardir))
T_N = 5


class DATracker(object):

    def __init__(self, image, region):


        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)
        #
        # right = min(region.x + region.width, image.shape[1] - 1)
        # bottom = min(region.y + region.height, image.shape[0] - 1)

        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)


        self.siam = SiameseNet(BaselineEmbeddingNet())
        # weights_init(siam)
        siamfc_path = project_path + "/models/siamfc_pretrained.pth"

        pretrained_siam = torch.load(siamfc_path)
        siam_dict = self.siam.state_dict()
        pretrained_siam = {k: v for k, v in pretrained_siam.items() if k in siam_dict}
        siam_dict.update(pretrained_siam)
        self.siam.load_state_dict(siam_dict)

        self.pi = T_Policy(T_N)

        pretrained_pi_dict = torch.load(project_path + '/models/Qnet/template_policy/10400_template_policy.pth')
        pi_dict = self.pi.state_dict()
        pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict}
        # pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict and k.startswith("conv")}
        pi_dict.update(pretrained_pi_dict)
        self.pi.load_state_dict(pi_dict)

        self.q = QNet_cir()  # .load_state_dict(torch.load("../Models/500_actor.pth"))
        pretrained_q_dict = torch.load(project_path + "/models/Qnet/QLT/10400_Qnet.pth")
        q_dict = self.q.state_dict()
        pretrained_q_dict = {k: v for k, v in pretrained_q_dict.items() if k in q_dict}
        q_dict.update(pretrained_q_dict)
        self.q.load_state_dict(q_dict)
        self.tracker = SiamFCTracker(model_path=siamfc_path, gpu_id=0)
        if torch.cuda.is_available():
            self.siam = self.siam.cuda()
            self.pi = self.pi.cuda()
            self.q = self.q.cuda()
        init_bbox = np.array([left, top, region.width, region. height])
        self.rate = init_bbox[2] / init_bbox[3]
        self.template = self.tracker.init(image, init_bbox)
        self.templates = []
        for i in range(T_N):
            self.templates.append(self.template)


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
        if action[0][action_id] * 0.9 > action[0][0]:
            template = self.templates[action_id]
        else:
            template = self.templates[0]
        with torch.no_grad():
            siam_box = self.tracker.update(image, template)
        siam_box = np.round([siam_box[0], siam_box[1], siam_box[2] -siam_box[0], siam_box[3] - siam_box[1]])
        bbox = siam_box
        for i in range(5):
            img_crop_l, _, _ = crop_image_actor_(np.array(np_img), bbox)
            # imo_crop_l = (np.array(img_crop_l).reshape(3, 107, 107))
            imo_l = np2tensor(np.array(img_crop_l).reshape(1, 107, 107, 3))
            deta_pos = np.zeros(3)
            with torch.no_grad():
                action_q = self.q.sample_action(imo_l)
            deta = 0.04
            if (action_q == 0):
                break
            if (action_q == 1):
                deta_pos[1] -= deta
            if (action_q == 2):
                deta_pos[1] += deta
            if (action_q == 3):
                deta_pos[0] -= deta
            if (action_q == 4):
                deta_pos[0] += deta
            if (action_q == 5):
                deta_pos[2] += deta
            if (action_q == 6):
                deta_pos[2] -= deta

            pos_ = np.round(move_crop_tracking(np.array(siam_box), deta_pos, (image.shape[1], image.shape[0]), self.rate))
            bbox = pos_
        result = bbox
        return vot.Rectangle(result[0], result[1], result[2], result[3])


handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
rett = dict()
rett["imagefile"] = imagefile
image = cv2.imread(imagefile, cv2.COLOR_BGR2RGB)
# print(image)
tracker = DATracker(image, selection)
while True:
    imagefile = handle.frame()
    rett["imagefile"] = imagefile
    if not imagefile:
        break
    image = cv2.imread(imagefile, cv2.COLOR_BGR2RGB)
    region = tracker.track(image)
    handle.report(region,rett)
