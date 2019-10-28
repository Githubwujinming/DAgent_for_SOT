import cv2
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,'../modules')
from actor import *
from options import *
from data_prov import *
from sample_generator import *

from modules.QNet_cir import QNet_cir
from modules.SiameseNet import SiameseNet
from modules.SiamFcTracker import SiamFCTracker
from modules.EmbeddingNet import BaselineEmbeddingNet
from modules.tem_policy_base import T_Policy, weights_init

from utils.np2tensor import np2tensor
from utils.init_video import _init_video
from utils.cal_distance import cal_distance
from utils.crop_image import crop_image_actor_
from utils.crop_image import move_crop_tracking
from utils.region_to_bbox import region_to_bbox
from utils.getbatch_actor import getbatch_actor
from utils.compile_results import _compile_results

T_N = 5
INTERVRAL = 10


def run_tracking(img_list, init_bbox, gt=None, savefig_dir='', display=False, siamfc_path = "../models/siamfc_pretrained.pth", gpu_id=0):
    rate = init_bbox[2] / init_bbox[3]
    target_bbox = np.array(init_bbox)

    siam = SiameseNet(BaselineEmbeddingNet())
    weights_init(siam)
    pretrained_siam = torch.load(siamfc_path)
    siam_dict = siam.state_dict()
    pretrained_siam = {k: v for k, v in pretrained_siam.items() if k in siam_dict}
    siam_dict.update(pretrained_siam)
    siam.load_state_dict(siam_dict)

    pi = T_Policy(T_N)
    # weights_init(policy)

    pretrained_pi_dict = torch.load('../models/Qnet/template_policy/10000_template_policy.pth')
    pi_dict = pi.state_dict()
    pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict}
    # pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict and k.startswith("conv")}
    pi_dict.update(pretrained_pi_dict)
    pi.load_state_dict(pi_dict)



    q = QNet_cir()
    pretrained_q_dict = torch.load("../models/Qnet/QLT{}_Qnet.pth".format(10000))
    q_dict = q.state_dict()
    pretrained_q_dict = {k: v for k, v in pretrained_q_dict.items() if k in q_dict}
    q_dict.update(pretrained_q_dict)
    q.load_state_dict(q_dict)

    tracker = SiamFCTracker(model_path=siamfc_path, gpu_id=gpu_id)
    if opts['use_gpu']:
        siam = siam.cuda()
        policy = pi.cuda()
        q = q.cuda()
        # tracker = tracker.cuda()

    image = cv2.cvtColor(cv2.imread(img_list[0]), cv2.COLOR_BGR2RGB)
    result = np.zeros((len(img_list), 4))
    result[0] = target_bbox

    spf_total = 0
    if display:
        dpi = 80.0
        figsize = (image.shape[1] / dpi, image.shape[0] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result[0, :2]), result[0, 2], result[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        siam_rect = plt.Rectangle(tuple(result[0, :2]), result[0, 2], result[0, 3],
                             linewidth=3, edgecolor="#0000ff", zorder=1, fill=False)
        ax.add_patch(rect)
        ax.add_patch(siam_rect)

        if display:
            plt.pause(.01)
            plt.draw()

    # deta_flag, out_flag_first = init_actor(actor, image, target_bbox)
    template = tracker.init(image, init_bbox)
    templates = []
    for i in range(T_N):
        templates.append(template)
    for frame in range(1, len(gt)):
        tic = time.time()
        # img = Image.open(frame_name_list[frame]).convert('RGB')
        cv2_img = cv2.cvtColor(cv2.imread(img_list[frame]), cv2.COLOR_BGR2RGB)
        np_img = np.array(cv2.resize(cv2_img, (255, 255),interpolation=cv2.INTER_AREA)).transpose(2, 0, 1)
        np_imgs = []
        for i in range(T_N):
            np_imgs.append(np_img)
        with torch.no_grad():
            responses = siam(torch.Tensor(templates).permute(0, 3, 1, 2).float().cuda(), torch.Tensor(np_imgs).float().cuda())
            action = policy(responses.permute(1, 0, 2, 3).cuda()).cpu().detach().numpy()
        action_id = np.argmax(action)
        if action[0][action_id] * 0.9 > action[0][0]:
            template = templates[action_id]
        else:
            template = templates[0]
        with torch.no_grad():
            siam_box = tracker.update(cv2_img, template)
        siam_box = np.round([siam_box[0], siam_box[1], siam_box[2] -siam_box[0], siam_box[3] - siam_box[1]])
        pos = siam_box
        for i in range(5):
            img_crop_l, _, _ = crop_image_actor_(np.array(cv2_img), pos)
        # imo_crop_l = (np.array(img_crop_l).reshape(3, 107, 107))
            imo_l = np2tensor(np.array(img_crop_l).reshape(1, 107, 107, 3))
            deta_pos = np.zeros(3)
            with torch.no_grad():
                action_q = q.sample_action(imo_l)
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


            pos_ = np.round(move_crop_tracking(np.array(siam_box), deta_pos, (image.shape[1], image.shape[0]), rate))
            pos = pos_
        result[frame] = pos_
        spf = time.time() - tic
        spf_total += spf

        if display:
            im.set_data(cv2_img)

            if gt is not None:
                gt_rect.set_xy(gt[frame, :2])
                gt_rect.set_width(gt[frame, 2])
                gt_rect.set_height(gt[frame, 3])

            rect.set_xy(result[frame, :2])
            rect.set_width(result[frame, 2])
            rect.set_height(result[frame, 3])

            siam_rect.set_xy(siam_box[:2])
            siam_rect.set_width(siam_box[2])
            siam_rect.set_height(siam_box[3])

            if display:
                plt.pause(.01)
                plt.draw()
        if frame % INTERVRAL == 0:
            template = tracker.init(cv2_img,gt[frame])
            # template = tracker.init(cv2_img, pos_* 0.5+ siam_box*0.5)
            templates.append(template)
            templates.pop(1)
    fps = len(img_list) / spf_total
    return result, fps


if __name__ == '__main__':

    img_path = '../dataset'
    savefig_dir = None
    siamfc_path = "../models/siamfc_pretrained.pth"
    video = 'Car4'
    display = 1

    gt, frame_name_list, _, _ = _init_video(img_path, video)
    ground_th = np.zeros([gt.shape[0], 4])
    for i in range(gt.shape[0]):
        ground_th[i] = region_to_bbox(gt[i], False)
    bboxes, fps = run_tracking(frame_name_list, gt[0], gt=gt, savefig_dir=savefig_dir, display=display)
    _, precision, precision_auc, iou = _compile_results(gt, bboxes, 20)
    print(video + \
          ' -- Precision ' + ': ' + "%.2f" % precision + \
          ' -- IOU: ' + "%.2f" % iou + \
          ' -- Speed: ' + "%.2f" % fps + ' --')