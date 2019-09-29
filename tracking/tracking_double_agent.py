import cv2
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,'../modules')
from actor import *
from options import *
from data_prov import *
from sample_generator import *

from modules.SiameseNet import SiameseNet
from modules.SiamFcTracker import SiamFCTracker
from modules.EmbeddingNet import BaselineEmbeddingNet
from modules.tem_policy_base import T_Policy, weights_init

from utils.init_video import _init_video
from utils.cal_distance import cal_distance
from utils.crop_image import move_crop_tracking
from utils.region_to_bbox import region_to_bbox
from utils.getbatch_actor import getbatch_actor


T_N = 5
INTERVRAL = 10

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
        if opts['show_train']:
            print("Iter %d, Loss %.10f"%(iter, loss.item()))
        if loss.item() < 0.0001:
            deta_flag = 0
            return deta_flag
        deta_flag = 1
    return deta_flag, out_flag_first


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

    policy = T_Policy(T_N)
    weights_init(policy)

    actor = Actor()  # .load_state_dict(torch.load("../Models/500_actor.pth"))
    pretrained_act_dict = torch.load("../models/Double_agent/11200_DA_actor.pth")
    actor_dict = actor.state_dict()
    pretrained_act_dict = {k: v for k, v in pretrained_act_dict.items() if k in actor_dict}
    actor_dict.update(pretrained_act_dict)
    actor.load_state_dict(actor_dict)

    tracker = SiamFCTracker(model_path=siamfc_path, gpu_id=gpu_id)
    if gpu_id != None:
        siam = siam.cuda()
        policy = policy.cuda()
        # tracker = tracker.cuda()

    image = cv2.cvtColor(cv2.imread(img_list[0]), cv2.COLOR_BGR2RGB)
    result = np.zeros((len(img_list), 4))
    result[0] = target_bbox


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

    deta_flag, out_flag_first = init_actor(actor, image, target_bbox)
    template = tracker.init(image, init_bbox)
    templates = []
    for i in range(T_N):
        templates.append(template)
    for frame in range(1, len(gt)):
        # img = Image.open(frame_name_list[frame]).convert('RGB')
        cv2_img = cv2.cvtColor(cv2.imread(img_list[frame]), cv2.COLOR_BGR2RGB)
        np_img = np.array(cv2.resize(cv2_img, (255, 255),interpolation=cv2.INTER_AREA)).transpose(2, 0, 1)
        np_imgs = []
        for i in range(T_N):
            np_imgs.append(np_img)

        responses = siam(torch.Tensor(templates).permute(0, 3, 1, 2).float().cuda(), torch.Tensor(np_imgs).float().cuda())
        action = policy(responses.permute(1, 0, 2, 3).cuda()).cpu().detach().numpy()
        action_id = np.argmax(action)
        template = templates[action_id]
        siam_box = tracker.update(cv2_img, template)
        siam_box = np.round([siam_box[0], siam_box[1], siam_box[2] -siam_box[0], siam_box[3] - siam_box[1]])
        img_g, img_l, out_flag = getbatch_actor(np.array(image), np.array(siam_box).reshape([1, 4]))

        deta_pos = actor(img_l, img_g)
        deta_pos = deta_pos.data.clone().cpu().numpy()
        if deta_pos[:, 2] > 0.05 or deta_pos[:, 2] < -0.05:
            deta_pos[:, 2] = 0
        if deta_flag or (out_flag and not out_flag_first):
            deta_pos[:, 2] = 0

        pos_ = np.round(move_crop_tracking(np.array(siam_box), deta_pos, (image.shape[1], image.shape[0]), rate))
        result[frame] = pos_


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
            template = tracker.init(cv2_img, pos_)
            templates.append(template)
            templates.pop(1)




if __name__ == '__main__':
    # img_path = '../dataset'
    # savefig_dir = None
    #
    # video = 'Car4'
    # display = 1
    #
    # gt, frame_name_list, _, _ = _init_video(img_path, video)
    # ground_th = np.zeros([gt.shape[0], 4])
    # for i in range(gt.shape[0]):
    #     ground_th[i] = region_to_bbox(gt[i], False)
    # bboxes, result_bb, fps = run_tracking(frame_name_list, gt[0], gt=gt, savefig_dir=savefig_dir, display=display)
    # _, precision, precision_auc, iou = _compile_results(gt, result_bb, 20)
    # print(video + \
    #       ' -- Precision ' + ': ' + "%.2f" % precision + \
    #       ' -- IOU: ' + "%.2f" % iou + \
    #       ' -- Speed: ' + "%.2f" % fps + ' --')

    img_path = '../dataset'
    savefig_dir = None
    siamfc_path = "../models/siamfc_pretrained.pth"
    video = 'Car4'
    display = 1

    gt, frame_name_list, _, _ = _init_video(img_path, video)
    ground_th = np.zeros([gt.shape[0], 4])
    for i in range(gt.shape[0]):
        ground_th[i] = region_to_bbox(gt[i], False)
    run_tracking(frame_name_list, gt[0], gt=gt, savefig_dir=savefig_dir, display=display, siamfc_path=siamfc_path)
