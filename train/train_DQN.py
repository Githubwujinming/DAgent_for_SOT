import gc
import cv2
import time
import torch
import buffer

sys.path.insert(0,'../modules')
from actor import *
from options import *
from data_prov import *
from sample_generator import *


from trainer import *
from data_prov import *
from utils.crop_image import move_crop
from utils.crop_image import crop_image
from utils.compute_iou import _compute_iou
from utils.np2tensor import np2tensor, npBN
from utils.cal_distance import cal_distance
from utils.crop_image import crop_image_actor_

from modules.sample_generator import *
from modules.SiameseNet import SiameseNet
from modules.SiamFcTracker import SiamFCTracker
from modules.EmbeddingNet import BaselineEmbeddingNet
from modules.tem_policy_base import T_Policy, weights_init
from modules.QNet_cir import QNet_cir, ReplayBuffer, QNet_train

MAX_EPISODES = 250000
MAX_BUFFER = 3000
MAX_TOTAL_REWARD = 300
T_N = 5
INTERVRAL = 10

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    batch_img_g, _, _ = getbatch_actor(np.array(image), actor_samples)
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
        feat = actor(batch_img_g[cur_idx])
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

def train(continue_epi=5600, policy_path="../models/Qnet/template_policy/{}_template_policy.pth",siamfc_path = "../models/siamfc_pretrained.pth",gpu_id=0):
    #强化学习样本存储空间
    ram = ReplayBuffer()
    #q-learning 网络
    q = QNet_cir()
    #q-leraning 目标网络
    q_target = QNet_cir()
    #优化器
    q_optimizer = torch.optim.Adam(q.parameters(), lr=0.0005)
    #siamfc跟踪器
    siamfc = SiamFCTracker(model_path=siamfc_path, gpu_id=gpu_id)
    #模板选择网络
    pi = T_Policy(T_N)
    weights_init(pi)

    if continue_epi > 0:
        #加载模板选择网络预训练权重
        pretrained_pi_dict = torch.load(policy_path.format(continue_epi))
        pi_dict = pi.state_dict()
        pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict}  # and k.startswith("conv")}
        pi_dict.update(pretrained_pi_dict)
        pi.load_state_dict(pi_dict)

        pretrained_q_dict = torch.load("../models/Qnet/QLT/{}_Qnet.pth".format(continue_epi))
        q_dict = q.state_dict()
        pretrained_q_dict = {k: v for k, v in pretrained_q_dict.items() if k in q_dict}
        q_dict.update(pretrained_q_dict)
        q.load_state_dict(q_dict)

    q_target.load_state_dict(q.state_dict())



    siam = SiameseNet(BaselineEmbeddingNet())
    # weights_init(siam)
    pretrained_siam = torch.load(siamfc_path)
    siam_dict = siam.state_dict()
    pretrained_siam = {k: v for k, v in pretrained_siam.items() if k in siam_dict}
    siam_dict.update(pretrained_siam)
    siam.load_state_dict(siam_dict)

    if torch.cuda.is_available():
        pi = pi.cuda()
        siam = siam.cuda()
        q = q.cuda()
        q_target = q_target.cuda()

    var = 0.3
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

        reward_all = 0
        templates = []
        for init_num in range(1):
            template = siamfc.init(img, ground_th)
            for i in range(T_N):
                templates.append(template)

        for frame in range(1, length):
            cv2_img = cv2.cvtColor(cv2.imread(frame_name_list[frame]), cv2.COLOR_BGR2RGB)
            np_img = np.array(cv2.resize(cv2_img, (255, 255), interpolation=cv2.INTER_AREA)).transpose(2, 0, 1)
            np_imgs = []
            for i in range(T_N):
                np_imgs.append(np_img)
            with torch.no_grad():
                responses = siam(torch.Tensor(templates).permute(0, 3, 1, 2).float().cuda(),
                                 torch.Tensor(np_imgs).float().cuda())
            pi_input = torch.tensor(responses).permute(1, 0, 2, 3).cuda()
            del responses, np_imgs, np_img
            action = pi(pi_input).cpu()


            action_id = np.argmax(action.detach().numpy())
            template = templates[action_id]
            with torch.no_grad():
                siam_box_oral = siamfc.update(cv2_img, templates[0])
                siam_box = siamfc.update(cv2_img, template)
            siam_box_oral = [siam_box_oral[0], siam_box_oral[1], siam_box_oral[2] - siam_box_oral[0], siam_box_oral[3] - siam_box_oral[1]]
            siam_box = [siam_box[0], siam_box[1], siam_box[2] - siam_box[0], siam_box[3] - siam_box[1]]

            img_crop_l = crop_image(np.array(cv2_img), siam_box_oral)
            imo_crop_l = (np.array(img_crop_l).reshape(3, 107, 107))
            imo_l = np2tensor(np.array(img_crop_l).reshape(1, 107, 107, 3))
            del img_crop_l
            expect = 0
            act_pos = np.zeros(7)
            a = np.random.randint(7)
            pos = np.array(siam_box_oral)
            deta = 5
            deta_pos = np.zeros(3)
            if np.random.random(1) < var or frame <= 3 or frame % 30 == 0:
                expect = 1
                deta_pos_ = cal_distance(np.vstack([pos, pos]), np.vstack([gt[frame], gt[frame]]))[0]
                a_ind = np.argmax(np.abs(deta_pos_))
                if(a_ind == 0):
                    if(deta_pos_[a_ind]>0):
                        a = 3
                    else:
                        a = 4
                if(a_ind == 1):
                    if (deta_pos_[a_ind] > 0):
                        a = 1
                    else:
                        a = 2
                if(a_ind == 2):
                    if (deta_pos_[a_ind] > 0):
                        a = 6
                    else:
                        a = 5
            else:
                a = q.sample_action(imo_l)
            del imo_l
            act_pos[a] = 1
            if(a == 1):
                deta_pos[1] += deta/siam_box_oral[3]
            if(a == 2):
                deta_pos[1] -= deta/siam_box_oral[3]
            if(a == 3):
                deta_pos[0] += deta/siam_box_oral[2]
            if(a == 4):
                deta_pos[0] -= deta/siam_box_oral[2]
            if(a == 5):
                deta_pos[2] -= deta/max(siam_box_oral[3], siam_box_oral[2])
            if(a == 6):
                deta_pos[2] += deta/max(siam_box_oral[3], siam_box_oral[2])
            pos_ = move_crop(pos, deta_pos, img_size, rate)
            img_crop_l_ = crop_image(np.array(cv2_img), pos_)
            imo_l_ = np.array(img_crop_l_).reshape(3, 107, 107)
            iou_siam_oral = _compute_iou(siam_box_oral, gt[frame])
            if iou_siam_oral < 0.2:
                continue
            iou_siam = _compute_iou(siam_box, gt[frame])
            iou_ac = _compute_iou(pos_, gt[frame])
            if iou_ac > iou_siam_oral:
                reward_ac = 1
            else:
                reward_ac = -1
            if iou_siam > iou_siam_oral:
                reward_t = 1
            else:
                reward_t = -1
            message = "iou_siam_oral: %2f, iou_siam: %2f, iou_ac: %2f ,expecte :%d\n"%(iou_siam_oral, iou_siam, iou_ac, expect)
            with open("../logs/iou.txt", "a", encoding='utf-8') as f:
                f.write(message)
            if reward_ac or reward_t and iou_siam_oral > 0.6:
                template = siamfc.init(cv2_img, pos_)
                templates.append(template)
                templates.pop(1)
            log_pi = torch.log(action[0, action_id])
            pi.put_data((reward_t, log_pi))
            ram.put((npBN(imo_crop_l),act_pos, reward_ac,npBN(imo_l_)))
            reward_all += reward_ac
        with open("../logs/iou.txt", "a", encoding='utf-8') as f:
            f.write('\n\n')
        if ram.size() >= 640:
            QNet_train(q, q_target, ram, q_optimizer)
        pi.train_policy()
        reward_100 += reward_all
        if train_step % 100 == 0 and train_step != 0:
            q_target.load_state_dict(q.state_dict())
            print("# of episode:{}, avg score : {:.1f}, buffer size:{}".format(train_step, reward_100/100, ram.size()))
            reward_100 = 0
        if train_step % 400 == 0 and train_step != 0:
            torch.save(q_target.state_dict(), '../models/Qnet/QLT/'+ str(train_step + continue_epi) + '_Qnet.pth')
            torch.save(pi.state_dict(), '../models/Qnet/template_policy/'+ str(train_step + continue_epi) + '_template_policy.pth')
            print("save model----{}".format(str(train_step + continue_epi)))
        if train_step % 10000 == 0:
            var = var * 0.95


if __name__ == '__main__':
    train()


