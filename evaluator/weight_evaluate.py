import os
import json
import numpy as np
from sklearn.metrics import r2_score
from dataset.info import DatasetInfo
from dataset.normalization import denormalize
import pycocotools.coco as coco
from .process_result_dict import initial_dict, caculate_mean, draw_stat_fig
from .utils import get_pred, get_gt, match
import torch
from scipy.spatial.distance import cdist

class Evaluator:
    def __init__(self, cfg, dataset_name=None):
        self.results = []
        self.trait = cfg.trait
        if dataset_name is None:
            dataset_name = cfg.val.dataset
        info = DatasetInfo.dataset_info[dataset_name]
        self.coco = coco.COCO(info['anno_dir'])
        self.keys = list(self.coco.catToImgs.keys())
        self.data = {}
        self.data = initial_dict(self.keys)
        self.result_dir = cfg.result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

    def evaluate(self, output, batch):
        gt_mask_t = batch.get("gt_mask_t")
        if gt_mask_t is None or gt_mask_t.numel() == 0:
            return
        gt_cts, gt_catids, gt_ts, gt_mask_ts, gt_imgid = get_gt(batch)
        pred_cts, pred_ts = get_pred(output)
        match_index = match(pred_cts, gt_cts)
        pred_ts = pred_ts[match_index[:, 0]]
        gt_ts = gt_ts[match_index[:, 1]]
        gt_catids = gt_catids[match_index[:, 1]]
        gt_mask_ts = gt_mask_ts[match_index[:, 1]]

        pred_ts = denormalize(pred_ts, self.trait)
        gt_ts = denormalize(gt_ts, self.trait)

        img_id = int(gt_imgid)
        name = self.coco.loadImgs(img_id)[0]['file_name']
        date = '/'.join(name.split('_')[4:6])

        dets = []
        for i in range(pred_ts.shape[0]):
            if gt_mask_ts[i] ==0.:
                continue
            result_eval = {
                'image_id': img_id,
                'date': date,
                'category_id': int(gt_catids[i]),
                'predict_trait': float(pred_ts[i]),
                'true_trait': float(gt_ts[i]),
            }
            dets.append(result_eval)

            if str(gt_catids[i]) not in self.data:
                self.data[str(gt_catids[i])] = {}
            if date not in self.data[str(gt_catids[i])]:
                self.data[str(gt_catids[i])][date] = {'predict': [], 'true': float(gt_ts[i])}
            elif 'true' not in self.data[str(gt_catids[i])][date]:
                self.data[str(gt_catids[i])][date]['true'] = float(gt_ts[i])

            self.data[str(gt_catids[i])][date]['predict'].append(float(pred_ts[i]))


        self.results.extend(dets)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        self.results = []

        global_stat = caculate_mean(self.data)
        json.dump(self.data, open(os.path.join(self.result_dir, 'stat_results.json'), 'w'))
        json.dump(global_stat, open(os.path.join(self.result_dir, 'global_stat_results.json'), 'w'))

        D = json.load(open(os.path.join(self.result_dir, 'stat_results.json'), 'r'))
        p = np.array([result['predict'] for PigId, day in D.items() for d, result in day.items() if len(result) != 0])
        t = np.array([result['true'] for PigId, day in D.items() for d, result in day.items() if len(result) != 0])
        mae = np.mean(np.abs(p - t))
        mse = np.mean(np.square(p - t))
        mape = np.mean(np.abs(p - t) / t)
        r2 = r2_score(t, p)
        print('MAE:{:.4f} , MSE:{:.4f} , MAPE:{:.4f}, R2:{:.4f}'.format(mae, mse, mape * 100, r2))
        fig_stat = {}
        for id, data in self.data.items():
            fig_stat[id] = draw_stat_fig(data, 'PIG{}'.format(id))
        fig_stat['Global'] = draw_stat_fig(global_stat, 'ALL')

        self.data = {}

        return {'MAE': mae, 'MSE': mse, 'MAPE': mape * 100, 'R2': r2}, fig_stat
