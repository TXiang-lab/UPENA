import torch
from scipy.spatial.distance import cdist
import numpy as np


def get_pred(o):
    valid = o['detection'][..., -1] < 0.9
    d = o['detection'][:][valid]
    wt = o['pred_ts'][valid]
    return d[:, :2].detach().cpu().numpy(), wt.detach().cpu().numpy()


def get_gt(b):
    ct_ind = b['ct_ind'][0]
    _, _, height, width = b['ct_hm'].size()
    ct_x, ct_y = ct_ind % width, ct_ind // width
    cts = torch.cat([ct_x.unsqueeze(-1), ct_y.unsqueeze(-1)], dim=-1).detach().cpu().numpy()
    catids = b['ct_cls'][0].detach().cpu().numpy()
    wts = b['gt_t'][0].detach().cpu().numpy()
    mask_wts = b['gt_mask_t'][0].detach().cpu().numpy()

    imgid = b['meta']['img_id'].detach().cpu().numpy()
    return cts, catids, wts,mask_wts, imgid


def match(pred_cts, gt_cts):
    gt_num = len(gt_cts)
    pred_num=len(pred_cts)
    D = cdist(pred_cts, gt_cts)
    pairs = []

    for i in range(gt_num):
        index = np.unravel_index(np.argmin(D), D.shape)
        D[index[0], :] = np.inf
        D[:, index[1]] = np.inf
        pairs.append(index)
        if i ==(pred_num-1):
            break
    return np.array(pairs)