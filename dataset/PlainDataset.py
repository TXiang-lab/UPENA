import math
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
from dataset.utils import transform_polys, filter_tiny_polys, get_cw_polys, gaussian_radius, draw_umich_gaussian, \
    augment
from dataset.normalization import normalize
import os
import cv2
import datetime

import json



class Dataset(data.Dataset):
    def __init__(self, anno_file, data_root, split, data_file, trait):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.trait = trait

        self.coco = COCO(anno_file)
        self.anns = np.array(sorted(self.coco.getImgIds()))
        self._initial_traits_dic(data_file)
        self._filter_anns()
    def _initial_traits_dic(self,file):
        self.data = json.load(open(file))
    def _filter_anns(self):
        cache = []
        dates = self.data['1'].keys()
        for a in self.anns:
            date = '/'.join(self.coco.loadImgs(int(a))[0]['file_name'].split('_')[4:6])
            if date in dates:
                cache.append(a)
        self.anns = cache

    def process_info(self, ann):
        image_id = ann
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=0)
        image_path = os.path.join(self.data_root, self.coco.loadImgs(int(image_id))[0]['file_name'])
        ann = self.coco.loadAnns(ann_ids)
        return ann, image_path, image_id


    def read_original_data(self, anno, image_path):
        img = cv2.imread(image_path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in instance['segmentation']] for instance in anno
                          if not isinstance(instance['segmentation'], dict)]
        cls_ids = [instance['category_id'] for instance in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]
            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = filter_tiny_polys(instance)
            polys = get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        # ct_hm = ct_hm[cls_id]
        ct_hm = ct_hm[0]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def get_trait(self, cls_id, filename):
        cls_id = str(cls_id)
        date = '/'.join(filename.split('_')[4:6])
        trt = float(self.data[cls_id][date])
        return trt

    def get_mask(self, poly, mask):
        cache = poly.reshape(1, -1, 2).astype('int32')
        _ = cv2.polylines(mask, cache, 1, 1)
        _ = cv2.fillPoly(mask, cache, 1)

    def __getitem__(self, index):
        data_input = {}
        ann = self.anns[index]
        anno, image_path, image_id = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, image_path)
        width, height = img.shape[1], img.shape[0]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split
            )
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)

        # detection
        output_h, output_w = inp_out_hw[2:]

        n_classes = 2  # fore+back
        ct_hm = np.zeros([n_classes, output_h, output_w], dtype=np.float32)
        semantic_mask = np.zeros([output_h, output_w], dtype='uint8')
        ct_cls = []
        wh = []
        ct_ind = []
        trait_data = []
        mask_trait = []

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                self.get_mask(poly, semantic_mask)
                t = self.get_trait(cls_id, image_path.split('/')[-1])
                if t == 0.:
                    mask_trait.append(0.)
                else:
                    mask_trait.append(1.)
                t = normalize(t, self.trait)

                trait_data.append(t)

        data_input.update({'inp': inp})
        detection = {'ct_hm': ct_hm,
                     'semantic_mask': semantic_mask.reshape(1, semantic_mask.shape[0], semantic_mask.shape[1]).astype(
                         np.float32), 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'gt_t': trait_data,
                     'gt_mask_t': mask_trait}
        data_input.update(detection)
        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': image_id, ' ann': ann, 'ct_num': ct_num}
        data_input.update({'meta': meta})
        return data_input

    def __len__(self):
        return len(self.anns)
