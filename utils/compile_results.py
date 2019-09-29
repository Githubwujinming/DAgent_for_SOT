import numpy as np
from utils.region_to_bbox import region_to_bbox
from utils.compute_distance import _compute_distance
from utils.compute_iou import _compute_iou


def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])
    precision = sum(new_distances < dist_threshold) * 1.0 / np.size(new_distances) * 100
    thresholds = np.linspace(0, 25, n_thresholds + 1)
    thresholds = thresholds[-n_thresholds:]
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i]) / np.size(new_distances)

    precision_auc = np.trapz(precisions_ths)

    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou
