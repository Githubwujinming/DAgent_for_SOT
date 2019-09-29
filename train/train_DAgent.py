import math
import gc
import torch
import buffer
import numpy as np
import cv2
from trainer import *
from data_prov import *
from modules.sample_generator import *
import time
from visdom import Visdom

from utils.cal_distance import cal_distance
from utils.getbatch_actor import getbatch_actor
from utils.crop_image import crop_image_actor_, crop_image
from utils.PILloader import loader
from utils.crop_image import move_crop
from utils.compute_iou import _compute_iou
from utils.np2tensor import np2tensor, npBN
from modules.tem_policy_base import T_Policy, weights_init
from modules.SiamFcTracker import SiamFCTracker
MAX_EPISODES = 250000
MAX_STEPS = 1000
MAX_BUFFER = 3000
MAX_TOTAL_REWARD = 300
T_N = 5
INTERVRAL = 10

def train(continue_epi=250000, policy_path="../models/template_policy/50000_base_policy.pth",siamfc_path = "../models/siamfc_pretrained.pth",gpu_id=0):
    ram = buffer.MemoryBuffer(MAX_BUFFER)
    siamfc = SiamFCTracker(model_path=siamfc_path, gpu_id=gpu_id)
    pi = T_Policy(T_N)
    weights_init(pi)
    pretrained_pi_dict = torch.load(policy_path)
    pi_dict = pi.state_dict()
    pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict and k.startswith("conv")}
    pi_dict.update(pretrained_pi_dict)
    pi.load_state_dict(pi_dict)


    if torch.cuda.is_available():
        pi = pi.cuda()
    ac_trainer = Trainer(ram)
    # continue_epi = 0
    if continue_epi > 0:
        ac_trainer.load_models(continue_epi)
    var = 0.3
    start_time = time.time()
    vis = Visdom(env='td_error')
    line_loss = vis.line(np.arange(1))
    train_ilsvrc_data_path = 'ilsvrc_train_new.json'
    ilsvrc_home = '/media/x/D/wujinming/ILSVRC2015_VID/ILSVRC2015/Data/VID'
    # ilsvrc_home = '/media/ubuntu/DATA/Document/ILSVRC2015_VID/ILSVRC2015/Data/VID'
    reward_100 = 0
    train_dataset = ILSVRCDataset(train_ilsvrc_data_path, ilsvrc_home + '/train')
    for train_step in range(MAX_EPISODES):
        frame_name_list, gt, length = train_dataset.next()
        img = cv2.cvtColor(cv2.imread(frame_name_list[0]), cv2.COLOR_BGR2RGB)
        img_size = (img.shape[1], img.shape[0])

        ground_th = gt[0]
        rate = ground_th[2] / ground_th[3]

        pos = ground_th
        reward_all = 0
        templates = []
        for init_num in range(1):
            ac_trainer.init_actor(img, ground_th)
            img = cv2.cvtColor(cv2.imread(frame_name_list[init_num]), cv2.COLOR_BGR2RGB)
            template = siamfc.init(img, ground_th)
            for i in range(T_N):
                templates.append(template)

        for frame in range(1, length):
            img = cv2.cvtColor(cv2.imread(frame_name_list[frame]), cv2.COLOR_BGR2RGB)
            responses = []

            pos_ = pos
            for i in range(T_N):
                template = templates[i]
                response = siamfc.response_map(img, template)
                # print(response.shape)
                responses.append(response[None, :, :])
            pi_input = torch.tensor(responses).permute(1, 0, 2, 3).cuda()
            action = pi(pi_input).cpu()
            action_id = np.argmax(action.detach().numpy())
            template = templates[action_id]
            siam_box_oral = siamfc.update(img, templates[0])
            siam_box_oral = [siam_box_oral[0], siam_box_oral[1], siam_box_oral[2] - siam_box_oral[0], siam_box_oral[3] - siam_box_oral[1]]
            siam_box = siamfc.update(img, template)
            siam_box = [siam_box[0], siam_box[1], siam_box[2] - siam_box[0], siam_box[3] - siam_box[1]]

            img_crop_l, img_crop_g, _ = crop_image_actor_(np.array(img), siam_box)
            imo_crop_l = (np.array(img_crop_l).reshape(3, 107, 107))
            imo_crop_g = (np.array(img_crop_g).reshape(3, 107, 107))

            imo_l = np2tensor(np.array(img_crop_l).reshape(1, 107, 107, 3))
            imo_g = np2tensor(np.array(img_crop_g).reshape(1, 107, 107, 3))



            # img_l = np2tensor(np_img_l)
            # torch_image = loader(img.resize((255, 255),Image.ANTIALIAS)).unsqueeze(0).cuda().mul(255.)
            deta_pos = ac_trainer.actor(imo_l, imo_g).squeeze(0).cpu().detach().numpy()

            if np.random.random(1) < var or frame <= 3 or frame % 20 == 0:
                deta_pos_ = cal_distance(np.vstack([pos, pos]), np.vstack([gt[frame], gt[frame]]))
                if np.max(abs(deta_pos_)) < 0.1:
                    deta_pos = deta_pos_[0]

            if deta_pos[2] > 0.1 or deta_pos[2] < -0.1:
                deta_pos[2] = 0

            pos_ = move_crop(pos_, deta_pos, img_size, rate)
            if frame % INTERVRAL == 0:
                template = siamfc.init(img, pos_)
                templates.append(template)
                templates.pop(1)
            img_crop_l_, img_crop_g_, out_flag = crop_image_actor_(np.array(img), pos_)
            # if out_flag:
            #     pos = gt[frame]
            #     continue
            imo_l_ = np.array(img_crop_l_).reshape(3, 107, 107)
            imo_g_ = np.array(img_crop_g_).reshape(3, 107, 107)

            # img_l_ = np.array(img_l_).reshape(1, 127, 127, 3)
            # gt_frame = gt[frame]
            iou_siam_oral = _compute_iou(siam_box_oral, gt[frame])
            iou_siam = _compute_iou(siam_box, gt[frame])
            iou_ac = _compute_iou(pos_, gt[frame])

            if iou_ac > iou_siam:
                reward_ac = 1
            else:
                reward_ac = -1
            if iou_siam > iou_siam_oral:
                reward_t = 1
            else:
                reward_t = -1

            log_pi = torch.log(action[0, action_id])
            pi.put_data((reward_t, log_pi))
            ac_trainer.ram.add(npBN(imo_crop_g), npBN(imo_g_), deta_pos, reward_ac, npBN(imo_crop_l), npBN(imo_l_))
            # if r == 0:
            #     break
            reward_all += reward_ac
            pos = pos_
            if out_flag or iou_ac == 0:
                pos = gt[frame]
        ac_trainer.optimize()
        pi.train_policy()
        reward_100 += reward_all
        gc.collect()
        if train_step % 100 == 0:
            td_error = ac_trainer.show_critic_loss()

            print(train_step, reward_100, 'td_error', td_error)
            y = np.array(td_error.cpu().detach().numpy())
            message = 'train_step: %d, reward_100: %d, td_error: %f \n' % (train_step, reward_100, y)
            with open("../logs/train_td_error.txt", "a", encoding='utf-8') as f:
                f.write(message)
            vis.line(X=np.array([train_step]), Y=np.array([y]),
                     win=line_loss,
                     update='append')
            reward_100 = 0

        if train_step % 200 == 0:
            ac_trainer.save_models(train_step)
            torch.save(pi.state_dict(), '../models/template_policy/'+ str(train_step) + '_template_policy.pth')
        if train_step % 10000 == 0:
            var = var * 0.95


if __name__ == '__main__':
    train()


