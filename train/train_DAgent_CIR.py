import gc
import cv2
import time
import torch
import buffer
from trainer import *
from data_prov import *
from visdom import Visdom

from utils.crop_image import move_crop
from utils.compute_iou import _compute_iou
from utils.np2tensor import np2tensor, npBN
from utils.cal_distance import cal_distance
from utils.crop_image import crop_image_actor_

from modules.sample_generator import *
from modules.SiameseNet import SiameseNet
from modules.SiamFcTracker import SiamFCTracker
from modules.EmbeddingNet import BaselineEmbeddingNet
from modules.tem_policy_base import T_Policy, weights_init

MAX_EPISODES = 250000
MAX_BUFFER = 3000
MAX_TOTAL_REWARD = 300
T_N = 5
INTERVRAL = 10

def train(continue_epi=247200, policy_path="../models/template_policy/{}_template_policy.pth",siamfc_path = "../models/siamfc_pretrained.pth",gpu_id=0):
    #强化学习样本存储空间
    ram = buffer.MemoryBuffer(MAX_BUFFER)
    ac_trainer = Trainer(ram)
    # continue_epi = 0
    if continue_epi > 0:
        policy_path = policy_path.format(continue_epi)
        ac_trainer.load_models(continue_epi)
    #siamfc跟踪器
    siamfc = SiamFCTracker(model_path=siamfc_path, gpu_id=gpu_id)
    #模板选择网络
    pi = T_Policy(T_N)
    weights_init(pi)
    pretrained_pi_dict = torch.load(policy_path)
    pi_dict = pi.state_dict()
    pretrained_pi_dict = {k: v for k, v in pretrained_pi_dict.items() if k in pi_dict}# and k.startswith("conv")}
    pi_dict.update(pretrained_pi_dict)
    pi.load_state_dict(pi_dict)

    siam = SiameseNet(BaselineEmbeddingNet())
    weights_init(siam)
    pretrained_siam = torch.load(siamfc_path)
    siam_dict = siam.state_dict()
    pretrained_siam = {k: v for k, v in pretrained_siam.items() if k in siam_dict}
    siam_dict.update(pretrained_siam)
    siam.load_state_dict(siam_dict)

    if torch.cuda.is_available():
        pi = pi.cuda()
        siam = siam.cuda()

    var = 0.5
    # vis = Visdom(env='td_error')
    # line_loss = vis.line(np.arange(1))
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
            pos_ = pos

            action_id = np.argmax(action.detach().numpy())
            template = templates[action_id]
            with torch.no_grad():
                siam_box_oral = siamfc.update(cv2_img, templates[0])
                siam_box = siamfc.update(cv2_img, template)
            siam_box_oral = [siam_box_oral[0], siam_box_oral[1], siam_box_oral[2] - siam_box_oral[0], siam_box_oral[3] - siam_box_oral[1]]
            siam_box = [siam_box[0], siam_box[1], siam_box[2] - siam_box[0], siam_box[3] - siam_box[1]]

            img_crop_l, img_crop_g, _ = crop_image_actor_(np.array(cv2_img), siam_box_oral)
            imo_crop_l = (np.array(img_crop_l).reshape(3, 107, 107))
            imo_crop_g = (np.array(img_crop_g).reshape(3, 107, 107))

            imo_l = np2tensor(np.array(img_crop_l).reshape(1, 107, 107, 3))
            imo_g = np2tensor(np.array(img_crop_g).reshape(1, 107, 107, 3))
            del img_crop_l, img_crop_g
            expect = 0
            deta_pos = ac_trainer.actor(imo_l, imo_g).squeeze(0).cpu().detach().numpy()
            del imo_l, imo_g
            if np.random.random(1) < var or frame <= 3 or frame % 20 == 0:
                deta_pos_ = cal_distance(np.vstack([pos, pos]), np.vstack([gt[frame], gt[frame]]))
                if np.max(abs(deta_pos_)) < 0.05:
                    expect = 1
                    deta_pos = deta_pos_[0]

            if deta_pos[2] > 0.05 or deta_pos[2] < -0.05:
                deta_pos[2] = 0

            pos_ = move_crop(np.array(siam_box_oral), deta_pos, img_size, rate)
            img_crop_l_, img_crop_g_, out_flag = crop_image_actor_(np.array(cv2_img), pos_)
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

            # reward_ac = iou_ac - iou_siam
            # reward_t = iou_siam - iou_siam_oral
            if iou_ac > iou_siam_oral:
                reward_ac = 1
            else:
                reward_ac = -1
            if iou_siam > iou_siam_oral:
                reward_t = 1
            else:
                reward_t = -1
            # print("iou_siam_oral: %2f, iou_siam: %2f, iou_ac: %2f"%(iou_siam_oral, iou_siam, iou_ac))
            message = "iou_siam_oral: %2f, iou_siam: %2f, iou_ac: %2f ,expecte :%d\n"%(iou_siam_oral, iou_siam, iou_ac, expect)
            # with open("../logs/iou.txt", "a", encoding='utf-8') as f:
            #     f.write(message)
            if reward_ac or reward_t and iou_siam_oral > 0.6:
                template = siamfc.init(cv2_img, pos_)
                templates.append(template)
                templates.pop(1)
            log_pi = torch.log(action[0, action_id])
            pi.put_data((reward_t, log_pi))
            ac_trainer.ram.add(npBN(imo_crop_g), npBN(imo_g_), deta_pos, reward_ac, npBN(imo_crop_l), npBN(imo_l_))
            # if r == 0:
            #     break
            reward_all += reward_ac
            pos = pos_
            if out_flag or iou_ac <= 0.2:
                pos = gt[frame]
        with open("../logs/iou.txt", "a", encoding='utf-8') as f:
            f.write('\n\n')
        ac_trainer.optimize()
        pi.train_policy()
        reward_100 += reward_all
        gc.collect()
        if train_step % 100 == 0 and train_step != 0:
            td_error = ac_trainer.show_critic_loss()

            print(train_step, reward_100, 'td_error', td_error)
            y = np.array(td_error.cpu().detach().numpy())
            message = 'train_step: %d, reward_100: %d, td_error: %f \n' % (train_step, reward_100, y)
            with open("../logs/train_td_error.txt", "a", encoding='utf-8') as f:
                f.write(message)
            # vis.line(X=np.array([train_step]), Y=np.array([y]),
            #          win=line_loss,
            #          update='append')
            reward_100 = 0

        if train_step % 400 == 0 and train_step != 0:
            ac_trainer.save_models(train_step)
            torch.save(pi.state_dict(), '../models/template_policy/'+ str(train_step + continue_epi) + '_template_policy.pth')
            print("save model----{}".format(str(train_step + continue_epi)))
        if train_step % 10000 == 0:
            var = var * 0.95


if __name__ == '__main__':
    train()


