import os
import numpy as np
from PIL import Image


def _init_video(img_path, video):
    if 'vot' in img_path:
        video_folder = os.path.join(img_path, video)
    else:
        video_folder = os.path.join(img_path, video, 'img')
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(video_folder, '') + s for s in frame_name_list]
    frame_name_list.sort()

    img = Image.open(frame_name_list[0])
    frame_sz = np.asarray(img.size)
    frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    if 'vot' in img_path:
        gt_file = os.path.join(video_folder, 'groundtruth.txt')
    else:
        gt_file = os.path.join(os.path.join(img_path, video), 'groundtruth_rect.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    if gt.shape.__len__() == 1:  # isnan(gt[0])
        gt = np.loadtxt(gt_file)
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames