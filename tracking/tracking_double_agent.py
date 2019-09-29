import sys
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0,'../modules')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *
from actor import *
import cv2
import numpy as np
import torch


np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)
torch.backends.cudnn.enabled = False

from utils.init_video import _init_video
from utils.region_to_bbox import region_to_bbox
from utils.crop_image import crop_image_blur
from utils.crop_image import move_crop
from utils.crop_image import move_crop_tracking, crop_image
from utils.cal_distance import cal_distance
from utils.PILloader import loader
from utils.getbatch_actor import getbatch_actor
from utils.compile_results import _compile_results
from utils.np2tensor import np2tensor

from modules.tem_policy_base import T_Policy, weights_init
from modules.SiameseNet import SiameseNet
from modules.EmbeddingNet import BaselineEmbeddingNet

def init_actor(actor, image, gt):
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)

    batch_num = 32
    maxiter = 80
    actor = actor.cuda()
    actor.train()
    init_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
    loss_func= torch.nn.MSELoss()
    _, _, out_flag_first = getbatch_actor(np.array(image), np.array(gt).reshape([1, 4]))
    actor_samples = np.round(gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, None),
                                         gt, 1500, [0.6, 1], [0.9, 1.1]))
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
        loss = loss_func(feat, Variable(torch.FloatTensor(batch_distance[cur_idx])).cuda())

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

def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

T_N = 5
def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)



    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while (len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()
        if opts['show_train']:
            print ("Iter %d, Loss %.10f" % (iter, loss.item()))


def forward_samples(model, image, samples, out_layer='conv3'):
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)

    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats

def run_tracking(img_list, init_bbox, gt=None, savefig_dir='', display=False):
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)
    siam = SiameseNet(BaselineEmbeddingNet())
    pi = T_Policy(T_N)
    if torch.cuda.is_available():
        siam = siam.cuda()
        pi = pi.cuda()
    templates = []
    img = Image.open(img_list[0]).convert('RGB')
    template = crop_image(img, init_bbox)
    for i in range(T_N):
        templates.append(template)

    rate = init_bbox[2] / init_bbox[3]
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox
    success = 1
    # Init model
    actor = Actor()#.load_state_dict(torch.load("../Models/500_actor.pth"))

    model = MDNet()#.load_state_dict(torch.load("../Models/500_critic.pth"))

    # pretrained_act_dict = torch.load("../Models/100_actor.pth")
    # pretrained_cri_dict = torch.load("../Models/100_critic.pth")
    #
    # actor_dict = actor.state_dict()
    # model_dict = model.state_dict()
    #
    # pretrained_act_dict = {k: v for k, v in pretrained_act_dict.items() if k in actor_dict}
    # pretrained_cri_dict = {k: v for k, v in pretrained_cri_dict.items() if k in model_dict and not k.startwith("branches")}
    #
    # actor_dict.update(pretrained_act_dict)
    # model_dict.update(pretrained_cri_dict)
    #
    # actor.load_state_dict(actor_dict)
    # model.load_state_dict(model_dict)

    if opts['use_gpu']:
        model = model.cuda()
        actor = actor.cuda()
    model.set_learnable_params(opts['fc_layers'])
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    image = Image.open(img_list[0]).convert('RGB')

    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)


    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])

    deta_flag, out_flag_first = init_actor(actor, image, target_bbox)

    init_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], 1, valid=False)
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=False)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]
    data_frame = [0]

    pos_score = forward_samples(model, image, np.array(init_bbox).reshape([1, 4]), out_layer='fc6')
    img_learn = [image]
    pos_learn = [init_bbox]
    score_pos = [pos_score.cpu().numpy()[0][1]]
    frame_learn = [0]
    pf_frame = []

    update_lenth = 10
    spf_total = 0
    # Display
    savefig = 0

    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)
    detetion = 0
    imageVar_first = cv2.Laplacian(crop_image_blur(np.array(image), target_bbox), cv2.CV_64F).var()
    for i in range(1, len(img_list)):
        cv2_img = cv2.cvtColor(cv2.imread(img_list[i]), cv2.COLOR_BGR2RGB)
        np_img = np.array(cv2.resize(cv2_img, (255, 255), interpolation=cv2.INTER_AREA)).transpose(2, 0, 1)
        np_imgs = []
        for i in range(T_N):
            np_imgs.append(np_img)
        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')
        if imageVar_first > 200:
            imageVar = cv2.Laplacian(crop_image_blur(np.array(image), target_bbox), cv2.CV_64F).var()
        else:
            imageVar = 200

        responses = siam(torch.Tensor(templates).permute(0, 3, 1, 2).float().cuda(),
                         torch.Tensor(np_imgs).float().cuda())
        action_tensor = pi(responses.permute(1, 0, 2, 3).cuda())
        del responses
        action = action_tensor.cpu().detach().numpy()
        action_id = np.argmax(action)
        template = templates[action_id]
        # Estimate target bbox
        img_g, img_l, out_flag = getbatch_actor(np.array(image), np.array(target_bbox).reshape([1, 4]))
        img_g = np2tensor(np.array(template).reshape(1, 107, 107, 3))
        # torch_image = loader(image.resize((255,255),Image.ANTIALIAS)).unsqueeze(0).cuda().add( - 128./255.)
        deta_pos = actor(img_l, img_g)
        del img_l, img_g
        deta_pos = deta_pos.data.clone().cpu().numpy()
        if deta_pos[:, 2] > 0.05 or deta_pos[:, 2] < -0.05:
            deta_pos[:, 2] = 0
        if deta_flag or (out_flag and not out_flag_first):
            deta_pos[:, 2] = 0
        if len(pf_frame) and i == (pf_frame[-1] + 1):
            deta_pos[:, 2] = 0

        pos_ = np.round(move_crop_tracking(target_bbox, deta_pos, (image.size[1], image.size[0]), rate))
        r = forward_samples(model, image, np.array(pos_).reshape([1, 4]), out_layer='fc6')
        r = r.cpu().numpy()

        if r[0][1] > 0 and imageVar > 100:
            target_bbox = pos_
            target_score = r[0][1]
            bbreg_bbox = pos_
            success = 1
            if True:
                fin_score = r[0][1]
                img_learn.append(image)
                pos_learn.append(target_bbox)
                score_pos.append(fin_score)
                frame_learn.append(i)
                while len(img_learn) > update_lenth * 2:
                    del img_learn[0]
                    del pos_learn[0]
                    del score_pos[0]
                    del frame_learn[0]
            result[i] = target_bbox
            result_bb[i] = bbreg_bbox
        else:
            detetion += 1
            if len(pf_frame) == 0:
                pf_frame = [i]
            else:
                pf_frame.append(i)

        if (len(frame_learn) == update_lenth * 2 and data_frame[-1] not in frame_learn) or data_frame[-1] == 0:
            for num in range(max(0, img_learn.__len__() - update_lenth), img_learn.__len__()):
                if frame_learn[num] not in data_frame:
                    gt_ = pos_learn[num]
                    image_ = img_learn[num]
                    pos_examples = np.round(gen_samples(pos_generator, gt_,
                                                        opts['n_pos_update'],
                                                        opts['overlap_pos_update']))
                    neg_examples = np.round(gen_samples(neg_generator, gt_,
                                                        opts['n_neg_update'],
                                                        opts['overlap_neg_update']))
                    pos_feats_ = forward_samples(model, image_, pos_examples)
                    neg_feats_ = forward_samples(model, image_, neg_examples)

                    pos_feats_all.append(pos_feats_)
                    neg_feats_all.append(neg_feats_)
                    data_frame.append(frame_learn[num])
                    if len(pos_feats_all) > 10:
                        del pos_feats_all[0]
                        del neg_feats_all[0]
                        del data_frame[0]
                else:
                    pos_feats_ = pos_feats_all[data_frame.index(frame_learn[num])]
                    neg_feats_ = neg_feats_all[data_frame.index(frame_learn[num])]

                if num == max(0, img_learn.__len__() - update_lenth):
                    pos_feats = pos_feats_
                    neg_feats = neg_feats_

                else:
                    pos_feats = torch.cat([pos_feats, pos_feats_], 0)
                    neg_feats = torch.cat([neg_feats, neg_feats_], 0)
            train(model, criterion, update_optimizer, pos_feats, neg_feats, opts['maxiter_update'])

        if success:
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            sample_generator.set_trans_f(opts['trans_f_expand'])

        if imageVar < 100:
            samples = gen_samples(init_generator, target_bbox, opts['n_samples'])
        else:
            samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])

            if i < 20 or ((init_bbox[2] * init_bbox[3]) > 1000 and (
                    target_bbox[2] * target_bbox[3] / (init_bbox[2] * init_bbox[3]) > 2.5 or target_bbox[2] *
                    target_bbox[3] / (init_bbox[2] * init_bbox[3]) < 0.4)):
                sample_generator.set_trans_f(opts['trans_f_expand'])
                samples_ = np.round(gen_samples(sample_generator, np.hstack(
                    [target_bbox[0:2] + target_bbox[2:4] / 2 - init_bbox[2:4] / 2, init_bbox[2:4]]), opts['n_samples']))
                samples = np.vstack([samples, samples_])

            sample_scores = forward_samples(model, image, samples, out_layer='fc6')
            top_scores, top_idx = sample_scores[:, 1].topk(5)
            top_idx = top_idx.cpu().numpy()
            target_score = top_scores.mean()
            target_bbox = samples[top_idx].mean(axis=0)
            success = target_score > opts['success_thr']

            # Bbox regression
            if success:
                bbreg_samples = samples[top_idx]
                bbreg_feats = forward_samples(model, image, bbreg_samples)
                bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
                bbreg_bbox = bbreg_samples.mean(axis=0)

                img_learn.append(image)
                pos_learn.append(target_bbox)
                score_pos.append(target_score)
                frame_learn.append(i)
                while len(img_learn) > 2 * update_lenth:
                    del img_learn[0]
                    del pos_learn[0]
                    del score_pos[0]
                    del frame_learn[0]

            else:
                bbreg_bbox = target_bbox

            # Copy previous result at failure
            if not success:
                target_bbox = result[i - 1]
                bbreg_bbox = result_bb[i - 1]

            # Save result
            result[i] = target_bbox
            result_bb[i] = bbreg_bbox

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '%04d.jpg' % (i)), dpi=dpi)
        if display:
            if gt is None:
                print
                "Frame %d/%d, Score %.3f, Time %.3f" % \
                (i, len(img_list), target_score, spf)
            else:
                if opts['show_train']:
                    print
                    "Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f, box (%d,%d,%d,%d), var %d" % \
                    (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf, target_bbox[0],
                     target_bbox[1], target_bbox[2], target_bbox[3], imageVar)

    fps = len(img_list) / spf_total
    return result, result_bb, fps



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
            bboxes, result_bb, fps = run_tracking(img_list, gt[0], gt=gt, savefig_dir=savefig_dir, display=0)
            _, precision, precision_auc, iou = _compile_results(gt, result_bb, 20)
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
        bboxes, result_bb, fps = run_tracking(frame_name_list, gt[0], gt=gt, savefig_dir=savefig_dir, display=display)
        _, precision, precision_auc, iou = _compile_results(gt, result_bb, 20)
        print (video + \
              ' -- Precision ' + ': ' + "%.2f" % precision + \
              ' -- IOU: ' + "%.2f" % iou + \
              ' -- Speed: ' + "%.2f" % fps + ' --')

