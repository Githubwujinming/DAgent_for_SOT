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
from modules.SiameseNet import SiameseNet
from modules.EmbeddingNet import BaselineEmbeddingNet
MAX_EPISODES = 250000
MAX_STEPS = 1000
MAX_BUFFER = 2000
MAX_TOTAL_REWARD = 300
T_N = 5
INTERVRAL = 10

def train(continue_epi=250000, policy_path="../Models/policy_template/50000_base_policy.pth"):
    ram = buffer.MemoryBuffer(MAX_BUFFER)
    siam = SiameseNet(BaselineEmbeddingNet())
    pi = T_Policy(T_N)
    pi.load_state_dict(torch.load(policy_path))
    if torch.cuda.is_available():
        pi = pi.cuda()
        siam = siam.cuda()
    trainer = Trainer(ram)
    # continue_epi = 0
    if continue_epi > 0:
        trainer.load_models(continue_epi)
    var = 0.5
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
        img = Image.open(frame_name_list[0]).convert('RGB')
        img_size = img.size

        ground_th = gt[0]
        rate = ground_th[2] / ground_th[3]

        pos = ground_th
        reward_all = 0
        templates = []
        for init_num in range(1):
            trainer.init_actor(img, ground_th)
            img = Image.open(frame_name_list[init_num]).convert('RGB')
            template = crop_image(np.array(img), ground_th)
            for i in range(T_N):
                templates.append(template)

        for frame in range(1, length):
            img = Image.open(frame_name_list[frame]).convert('RGB')
            pos_ = pos
            img_crop_l, img_crop_g, _ = crop_image_actor_(np.array(img), pos)
            imo_crop_l = (np.array(img_crop_l).reshape(3, 107, 107))
            imo_crop_g = (np.array(img_crop_g).reshape(3, 107, 107))

            imo_l = np2tensor(np.array(img_crop_l).reshape(1, 107, 107, 3))
            imo_g = np2tensor(np.array(img_crop_g).reshape(1, 107, 107, 3))

            cv2_img = cv2.cvtColor(cv2.imread(frame_name_list[frame]), cv2.COLOR_BGR2RGB)
            np_img = np.array(cv2.resize(cv2_img, (255, 255), interpolation=cv2.INTER_AREA)).transpose(2, 0, 1)
            np_imgs = []
            for i in range(T_N):
                np_imgs.append(np_img)
            responses = siam(torch.Tensor(templates).permute(0, 3, 1, 2).float().cuda(),
                             torch.Tensor(np_imgs).float().cuda())

            action_tensor = pi(responses.permute(1, 0, 2, 3).cuda())
            del responses
            action = action_tensor.cpu().detach().numpy()
            action_id = np.argmax(action)
            template = templates[action_id]
            imo_g = np2tensor(np.array(template).reshape(1, 107, 107, 3))

            # img_l = np2tensor(np_img_l)
            # torch_image = loader(img.resize((255, 255),Image.ANTIALIAS)).unsqueeze(0).cuda().mul(255.)
            deta_pos = trainer.actor(imo_l, imo_g).squeeze(0).cpu().detach().numpy()

            if np.random.random(1) < var or frame <= 5 or frame % 15 == 0:
                deta_pos_ = cal_distance(np.vstack([pos, pos]), np.vstack([gt[frame], gt[frame]]))
                if np.max(abs(deta_pos_)) < 0.1:
                    deta_pos = deta_pos_[0]

            if deta_pos[2] > 0.05 or deta_pos[2] < -0.05:
                deta_pos[2] = 0

            pos_ = move_crop(pos_, deta_pos, img_size, rate)
            if frame % INTERVRAL == 0:
                template = crop_image(np.array(img), pos_)
                templates.append(template)
                templates.pop(1)
            img_crop_l_, img_crop_g_, out_flag = crop_image_actor_(np.array(img), pos_)
            # if out_flag:
            #     pos = gt[frame]
            #     continue
            imo_l_ = np.array(img_crop_l_).reshape(3, 107, 107)
            imo_g_ = np.array(img_crop_g_).reshape(3, 107, 107)

            # img_l_ = np.array(img_l_).reshape(1, 127, 127, 3)
            gt_frame = gt[frame]
            r = _compute_iou(pos_, gt[frame])

            if r > 0.7:
                reward = 1
            elif r >= 0.5 and r <= 0.7:
                gt_pre = gt[frame - 1]
                r_pre = _compute_iou(pos, gt_pre)
                reward = max(0, r - r_pre)
            else:
                reward = -1
            imo_g_ = np.array(template).reshape(3, 107, 107)
            trainer.ram.add(npBN(imo_crop_g), npBN(imo_g_), deta_pos, reward, npBN(imo_crop_l), npBN(imo_g_))
            # if r == 0:
            #     break
            reward_all += reward
            pos = pos_
            if out_flag or r == 0:
                pos = gt[frame]
        trainer.optimize()
        reward_100 += reward_all
        gc.collect()
        if train_step % 100 == 0:
            td_error = trainer.show_critic_loss()

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
            trainer.save_models(train_step)
        if train_step % 10000 == 0:
            var = var * 0.95


if __name__ == '__main__':
    train()


