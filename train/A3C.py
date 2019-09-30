import cv2
import torch
import numpy as np
import time, threading
import torch.nn.functional as F

from data_prov import *
from model import Actor, Critic

from modules.SiameseNet import SiameseNet
from modules.SiamFcTracker import SiamFCTracker
from modules.EmbeddingNet import BaselineEmbeddingNet
from modules.tem_policy_base import T_Policy, weights_init

from utils.crop_image import move_crop
from utils.compute_iou import _compute_iou
from utils.np2tensor import np2tensor, npBN
from utils.cal_distance import cal_distance
from utils.crop_image import crop_image_actor_

T_N = 5

GAMMA = 0.99

RUN_TIME = 30

EPS_START = 0.4
EPS_STOP = .15
EPS_sTEPS = 75000

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

OPTIMIZERS = 2
THREADS = 8

LOSS_V = .5
LOSS_ENTROPY = .01

MIN_BATCH = 32

LEARNING_RATE = 5e-3
use_gpu = 1
class Brain:
    def __init__(self):
        self.pi_queue = [[], []]
        self.train_queue = [[], [], [], [], [], [], []]
        self.lock_queue = threading.Lock()
        self.actor = Actor
        self.critic = Critic
        self.pi = T_Policy(T_N)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.actor = torch.nn.DataParallel(self.actor)
            self.critic = torch.nn.DataParallel(self.critic)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), LEARNING_RATE)
        self.optimizer = torch.optim.Adam([self.actor.parameters(), self.critic.parameters()], LEARNING_RATE)
    def predict_q(self,s_l, a):
        s_l = torch.from_numpy(s_l)
        a = torch.from_numpy(a)
        if use_gpu:
            s_l = s_l.cuda()
            a = a.cuda()
        q = self.critic(s_l, a)

        if use_gpu:
            q = q.cpu().detach().numpy()
        else:
            q = q.detach().numpy()
        return q

    def predict_a(self, s_l, s_g):
        s_l = torch.from_numpy(s_l)
        s_g = torch.from_numpy(s_g)
        if use_gpu:
            s_l = s_l.cuda()
            s_g = s_g.cuda()
        action = self.actor(s_l, s_g)

        if use_gpu:
            action = action.cpu().detach().numpy()
        else:
            action = action.detach().numpy()
        return action

    def train_push(self, s_l, s_g, a, r, s_l_, s_g_, out, pi_r, log_pi):
        with self.lock_queue:
            self.train_queue[0].append(s_l)
            self.train_queue[1].append(s_g)
            self.train_queue[2].append(a)
            self.train_queue[3].append(r)
            self.train_queue[4].append(s_l_)
            self.train_queue[5].append(s_g_)
            self.train_queue[6].append(out)
            self.pi_queue[0].append(pi_r)
            self.pi_queue[1].append(log_pi)

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)
            return
        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:
                return
            s_l, s_g, a, r, s_l_, s_g_, out = self.train_queue
            self.train_queue = [[], [], [], [], [], [], []]

        s_g = np.vstack(s_g)
        s_g_ = np.vstack(s_g_)
        a = np.vstack(a)
        r = np.vstack(r)
        s_l = np.vstack(s_l)
        s_l_ = np.vstack(s_l_)
        out = np.vstack(out)
        if len(s_g) > 5 * MIN_BATCH: print ("Optimizer alert! Minimizing batch of %d" % len(s_g))

        a_ = self.predict_a(s_l_,s_g_)
        q = self.predict_q(s_l_, a_)
        r_expected = r + GAMMA_N * q * out

        r_predicted = self.predict_q(s_l, a)
        loss_critic = F.smooth_l1_loss(r_predicted, r_expected)

        pred_a1 = self.actor.forward(s_l, s_g)
        loss_actor = -1 * torch.sum(self.critic.forward(s_l, pred_a1))
        entropy = LOSS_ENTROPY * torch.reduce_sum(pred_a1 * torch.log(pred_a1 + 1e-10))
        loss_total = torch.reduce_mean(loss_actor + loss_critic * LOSS_V + entropy)
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        pi_R = 0
        for pi_r, log_prob in self.pi_queue[::-1]:
            pi_R = pi_r + GAMMA * pi_R
            loss = -log_prob * (pi_R - 0.3)
            self.pi_optimizer.zero_grad()
            loss.backward()
            self.pi_optimizer.step()
        self.pi_queue = [[], []]


frames = 0
class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.memory = []
        self.R = 0

    def getEpsilon(self):
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps

    def act(self, s_l, s_g):
        eps = self.getEpsilon()
        global frames;
        frames += 1
        action = brain.predict_a(s_l, s_g)
        return action

    def train(self, s_l, s_g, a, r, s_l_, s_g_, out, pi_r, log_pi):
        def get_sample(memory, n):
            s_l, s_g, a, _, _, _, _ = memory[0]
            _, _, _, _, s_l_, s_g_, out = memory[n-1]

            return s_l, s_g, a, self.R, s_l_, s_g_, out
        self.memory.append(s_l, s_g, a, r, s_l_, s_g_, out)

        self.R = (self.R + r * GAMMA_N) / GAMMA
        if out > 0:
            while len(self.memory) > 0:
                n = len(self.memory)
                s_l, s_g, a, r, s_l_, s_g_, out = get_sample((self.memory, n))
                brain.train_push(s_l, s_g, a, r, s_l_, s_g_, out, pi_r, log_pi)
                self.R = (self.R - self.memory[0][3]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s_l, s_g, a, r, s_l_, s_g_, out = get_sample((self.memory, N_STEP_RETURN))
            brain.train_push(s_l, s_g, a, r, s_l_, s_g_, out, pi_r, log_pi)
            self.R = (self.R - self.memory[0][3])
            self.memory.pop(0)


class Environment(threading.Thread):
    stop_signal = False

    def __init__(self,train_dataset, eps_start=EPS_START, eps_end=EPS_STOP, eps_step=EPS_sTEPS):
        threading.Thread.__init__(self)
        self.agent = Agent(eps_start, eps_end, eps_step)

        self.train_dataset = train_dataset

    def runEpisode(self):
        frame_name_list, gt, length = train_dataset.next()
        img = cv2.cvtColor(cv2.imread(frame_name_list[0]), cv2.COLOR_BGR2RGB)
        img_size = (img.shape[1], img.shape[0])

        ground_th = gt[0]
        rate = ground_th[2] / ground_th[3]

        pos = ground_th
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
            action = self.agent.pi(pi_input).cpu()
            pos_ = pos

            action_id = np.argmax(action.detach().numpy())
            template = templates[action_id]
            with torch.no_grad():
                siam_box_oral = siamfc.update(img, templates[0])
                siam_box = siamfc.update(img, template)
            siam_box_oral = [siam_box_oral[0], siam_box_oral[1], siam_box_oral[2] - siam_box_oral[0],
                             siam_box_oral[3] - siam_box_oral[1]]
            siam_box = [siam_box[0], siam_box[1], siam_box[2] - siam_box[0], siam_box[3] - siam_box[1]]

            img_crop_l, img_crop_g, _ = crop_image_actor_(np.array(img), siam_box)
            img_crop_l = (np.array(img_crop_l).reshape(3, 107, 107))
            img_crop_g = (np.array(img_crop_g).reshape(3, 107, 107))
            s_l, s_g = npBN(img_crop_l), npBN(img_crop_g)
            deta_pos = self.agent.act(s_l, s_g)
            if np.random.random(1) < var or frame <= 3 or frame % 20 == 0:
                deta_pos_ = cal_distance(np.vstack([pos, pos]), np.vstack([gt[frame], gt[frame]]))
                if np.max(abs(deta_pos_)) < 0.1:
                    deta_pos = deta_pos_[0]

            if deta_pos[2] > 0.05 or deta_pos[2] < -0.05:
                deta_pos[2] = 0
            pos_ = move_crop(pos_, deta_pos, img_size, rate)
            img_crop_l_, img_crop_g_, out_flag = crop_image_actor_(np.array(img), pos_)
            imo_l_ = np.array(img_crop_l_).reshape(3, 107, 107)
            imo_g_ = np.array(img_crop_g_).reshape(3, 107, 107)
            s_l_ = npBN(imo_l_)
            s_g_ = npBN(imo_g_)
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
            if reward_ac and reward_t and iou_siam_oral > 0.6:
                template = siamfc.init(img, pos_)
                templates.append(template)
                templates.pop(1)
            out = 1
            if out_flag or iou_ac <= 0.2:
                pos = gt[frame]
                out = 0
            log_pi = torch.log(action[0, action_id])
            self.agent.train(s_l, s_g, deta_pos, reward_ac, s_l_, s_g_, out, reward_t, log_pi)
            reward_all += reward_ac

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True


class Optimizer(threading.Thread):

    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


if __name__ == '__main__':
    policy_path = "../models/template_policy/11200_template_policy.pth"
    siamfc_path = "../models/siamfc_pretrained.pth"
    gpu_id = 0
    var = 0.4
    siamfc = SiamFCTracker(model_path=siamfc_path, gpu_id=gpu_id)
    # 模板选择网络

    siam = SiameseNet(BaselineEmbeddingNet())
    weights_init(siam)
    pretrained_siam = torch.load(siamfc_path)
    siam_dict = siam.state_dict()
    pretrained_siam = {k: v for k, v in pretrained_siam.items() if k in siam_dict}
    siam_dict.update(pretrained_siam)
    siam.load_state_dict(siam_dict)

    if torch.cuda.is_available():
        siam = siam.cuda()


    train_ilsvrc_data_path = 'ilsvrc_train_new.json'
    ilsvrc_home = '/media/x/D/wujinming/ILSVRC2015_VID/ILSVRC2015/Data/VID'
    # ilsvrc_home = '/media/ubuntu/DATA/Document/ILSVRC2015_VID/ILSVRC2015/Data/VID'
    train_dataset = ILSVRCDataset(train_ilsvrc_data_path, ilsvrc_home + '/train')
    env_test = Environment(train_dataset,eps_start=0., eps_end=0.)
    brain = Brain()
    envs = [Environment() for i in range(THREADS)]
    opts = [Optimizer() for i in range(OPTIMIZERS)]

    for o in opts:
        o.start()

    for e in envs:
        e.start()

    time.sleep(RUN_TIME)

    for e in envs:
        e.stop()

    for e in envs:
        e.join()

    for o in opts:
        o.stop()

    for o in opts:
        o.join()

    print("Training finished")
    env_test.run()