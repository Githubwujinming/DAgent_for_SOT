import os
import cv2
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0,'../modules')
from actor import *
from options import *
from data_prov import *
from sample_generator import *


np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)
# torch.backends.cudnn.enabled = False

from utils.PILloader import loader
from utils.crop_image import move_crop
from utils.init_video import _init_video
from utils.cal_distance import cal_distance
from utils.crop_image import crop_image_blur
from utils.overlap_ratio import overlap_ratio
from utils.crop_image import move_crop_tracking
from utils.region_to_bbox import region_to_bbox
from utils.getbatch_actor import getbatch_actor
from utils.compile_results import _compile_results

from modules.SiameseNet import SiameseNet
from modules.tem_policy_base import T_Policy, weights_init
from modules.SiamFcTracker import SiamFCTracker
from modules.EmbeddingNet import BaselineEmbeddingNet


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


def run_tracking(img_list, init_bbox, gt=None, savefig_dir='', display=False, siamfc_path = "../models/siamfc_pretrained.pth", policy_path="../models/template_policy/11200_template_policy.pth", gpu_id=0):

    rate = init_bbox[2] / init_bbox[3]
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    # result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    # result_bb[0] = target_bbox
    success = 1
    actor = Actor()#.load_state_dict(torch.load("../Models/500_actor.pth"))

    pretrained_act_dict = torch.load("../models/Double_agent/11200_DA_actor.pth")

    actor_dict = actor.state_dict()

    pretrained_act_dict = {k: v for k, v in pretrained_act_dict.items() if k in actor_dict}

    actor_dict.update(pretrained_act_dict)

    actor.load_state_dict(actor_dict)

    siamfc = SiamFCTracker(model_path=siamfc_path, gpu_id=gpu_id)
    siamEmbed = siam = SiameseNet(BaselineEmbeddingNet())
    T_N = opts['T_N']
    pi = T_Policy(T_N)
    weights_init(pi)
    # pretrained_pi_dict = torch.load(policy_path)
    # pi_dict = pi.state_dict()
    # pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict}
    # # pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict and k.startswith("conv")}
    # pi_dict.update(pretrained_pi_dict)
    # pi.load_state_dict(pi_dict)


    if opts['use_gpu']:
        actor = actor.cuda()
        siamEmbed = siamEmbed.cuda()
        pi = pi.cuda()

    image = cv2.cvtColor(cv2.imread(img_list[0]), cv2.COLOR_BGR2RGB)
    #init

    deta_flag, out_flag_first = init_actor(actor, image, target_bbox)
    template = siamfc.init(image, target_bbox)
    # t = template
    templates = []
    for i in range(T_N):
        templates.append(template)
    spf_total = 0
    # Display
    savefig = 0

    if display or savefig:
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
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)
    imageVar_first = cv2.Laplacian(crop_image_blur(np.array(image), target_bbox), cv2.CV_64F).var()
    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        image = cv2.cvtColor(cv2.imread(img_list[i]), cv2.COLOR_BGR2RGB)
        np_img = np.array(cv2.resize(image, (255, 255), interpolation=cv2.INTER_AREA)).transpose(2, 0, 1)
        np_imgs = []
        for i in range(T_N):
            np_imgs.append(np_img)
        if imageVar_first > 200:
            imageVar = cv2.Laplacian(crop_image_blur(np.array(image), target_bbox), cv2.CV_64F).var()
        else:
            imageVar = 200

        if opts['use_gpu']:
            responses = siamEmbed(torch.Tensor(templates).permute(0, 3, 1, 2).float().cuda(), torch.Tensor(np_imgs).float().cuda())
        else:
            responses = siamEmbed(torch.Tensor(templates).permute(0, 3, 1, 2).float(), torch.Tensor(np_imgs).float())
        # responses = []
        # for i in range(T_N):
        #     template = templates[i]
        #     response = siamfc.response_map(image, template)
        #     responses.append(response[None,:,:])
        if opts['use_gpu']:
            pi_input = torch.Tensor(responses.cpu()).permute(1, 0, 2, 3).cuda()
            action = pi(pi_input).cpu().detach().numpy()
        else:
            pi_input = torch.Tensor(responses).permute(1, 0, 2, 3)
            action = pi(pi_input).numpy()
        action_id = np.argmax(action)
        template = templates[action_id]
        siam_box = siamfc.update(image,templates[0])
        siam_box = np.round([siam_box[0], siam_box[1], siam_box[2] - siam_box[0], siam_box[3] - siam_box[1]])
        print(siam_box)
        # Estimate target bbox
        img_g, img_l, out_flag = getbatch_actor(np.array(image), np.array(siam_box).reshape([1, 4]))
        deta_pos = actor(img_l, img_g)
        deta_pos = deta_pos.data.clone().cpu().numpy()
        if deta_pos[:, 2] > 0.05 or deta_pos[:, 2] < -0.05:
            deta_pos[:, 2] = 0
        if deta_flag or (out_flag and not out_flag_first):
            deta_pos[:, 2] = 0

        pos_ = np.round(move_crop_tracking(np.array(siam_box), deta_pos, (image.shape[1], image.shape[0]), rate))

        if imageVar > 100:
            target_bbox = pos_
            result[i] = target_bbox
        if i % 10 == 0:
            template = siamfc.init(image, pos_)
            templates.append(template)
            templates.pop(1)

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result[i, :2])
            rect.set_width(result[i, 2])
            rect.set_height(result[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '%04d.jpg' % (i)), dpi=dpi)
        if display:
            if gt is None:
                print
                ("Frame %d/%d,  Time %.3f" % \
                (i, len(img_list), spf))
            else:
                if opts['show_train']:
                    print
                    ("Frame %d/%d, Overlap %.3f, Time %.3f, box (%d,%d,%d,%d), var %d" % \
                    (i, len(img_list), overlap_ratio(gt[i], result[i])[0], spf, target_bbox[0],
                     target_bbox[1], target_bbox[2], target_bbox[3], imageVar))

    fps = len(img_list) / spf_total
    return result, fps



if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--seq', default='DragonBaby', help='input seq')
    # parser.add_argument('-j', '--json', default='cfg.josn', help='input json')
    # parser.add_argument('-f', '--savefig', action='store_true')
    # parser.add_argument('-d', '--display', action='store_true')
    #
    # args = parser.parse_args()
    # assert (args.seq != '' or args.json != '')

    img_path = '../dataset'

    savefig_dir = None

    video = 'Car4'
    display = 1

    if video == 'all':
        opts['show_train'] = 0
        dataset_folder = os.path.join(img_path)
        videos_list = [v for v in os.listdir(dataset_folder)]
        videos_list.sort()

        nv = np.size(videos_list)
        speed_all = np.zeros(nv)
        precisions_all = np.zeros(nv)
        precisions_auc_all = np.zeros(nv)
        ious_all = np.zeros(nv)
        for i in range(nv):
            gt, img_list, _, _ = _init_video(img_path, videos_list[i])
            ground_th = np.zeros([gt.shape[0], 4])

            for video_num in range(gt.shape[0]):
                ground_th[video_num] = region_to_bbox(gt[video_num], False)
            bboxes, fps = run_tracking(img_list, gt[0], gt=gt, savefig_dir=savefig_dir, display=0)
            _, precision, precision_auc, iou = _compile_results(gt, bboxes, 20)
            speed_all[i] = fps
            precisions_all[i] = precision
            precisions_auc_all[i] = precision_auc
            ious_all[i] = iou

            print (str(i) + ' -- ' + videos_list[i] + \
                  ' -- Precision: ' + "%.2f" % precisions_all[i] + \
                  ' -- IOU: ' + "%.2f" % ious_all[i] + \
                  ' -- Speed: ' + "%.2f" % speed_all[i] + ' --')

        mean_precision = np.mean(precisions_all)
        mean_precision_auc = np.mean(precisions_auc_all)
        mean_iou = np.mean(ious_all)
        mean_speed = np.mean(speed_all)
        print ('-- Overall stats (averaged per frame) on ' + str(nv))
        print (' -- Precision ' + "(20 px)" + ': ' + "%.2f" % mean_precision + \
              ' -- IOU: ' + "%.2f" % mean_iou + \
              ' -- Speed: ' + "%.2f" % mean_speed + ' --')
        for i in range(len(videos_list)):
            print (round(precisions_all[i], 2))
        print (round(mean_precision, 2))
        for i in range(len(videos_list)):
            print (round(ious_all[i], 2))
        print (round(mean_iou, 2))

    else:

        gt, frame_name_list, _, _ = _init_video(img_path, video)
        ground_th = np.zeros([gt.shape[0], 4])
        for i in range(gt.shape[0]):
            ground_th[i] = region_to_bbox(gt[i], False)
        bboxes, fps = run_tracking(frame_name_list, gt[0], gt=gt, savefig_dir=savefig_dir, display=display)
        _, precision, precision_auc, iou = _compile_results(gt, bboxes, 20)
        print (video + \
              ' -- Precision ' + ': ' + "%.2f" % precision + \
              ' -- IOU: ' + "%.2f" % iou + \
              ' -- Speed: ' + "%.2f" % fps + ' --')

