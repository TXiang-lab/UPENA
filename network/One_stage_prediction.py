import torch
import torch.nn as nn
from network.backbone.Resnet import resnet18

from network.neck.FeaturePyramidNetworks import fpn
from network.head.ProtoNet import proto_net
from network.head.RegressionNet import coeff_net
import torch.nn.functional as F
from network.utils import decode_ct_hm
import numpy as np


class OneStageNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet18()

        self.backbone.channels = [256,512,1024,2048]

        self.fpn = fpn(self.backbone.channels)

        self.proto_net, self.mask_dim = proto_net(self.backbone.channels[0])
        self.coeff_net = coeff_net(self.backbone.channels[0], self.mask_dim)
        self.semantic_conv = nn.Conv2d(self.backbone.channels[0], 1, 1)
        self.predict = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(72 * 96, 256),
            nn.LayerNorm(256, elementwise_affine=False),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def use_gt_detection(self, detect_output, proto_output, batch):
        coeff = detect_output['coeff']
        ct_01 = batch['ct_01'].bool()
        ct_ind = batch['ct_ind'][ct_01]
        ct_img_idx = batch['ct_img_idx'][ct_01]
        _, _, height, width = batch['ct_hm'].size()
        ct_x, ct_y = ct_ind % width, ct_ind // width

        reg_coeff = coeff[ct_img_idx, :, ct_y, ct_x].unsqueeze(-1).unsqueeze(-1)
        proto = proto_output[ct_img_idx, ...]
        return reg_coeff, proto

    def forward(self, x, batch=None):
        outs = self.backbone(x)
        outs = self.fpn(outs)[0]
        proto_out = self.proto_net(outs)
        detect_outs = self.coeff_net(outs)

        if self.training:
            coeffs, protos = self.use_gt_detection(detect_outs, proto_out, batch)
            fm = (protos * coeffs).sum(dim=1)
            pred_wts = self.predict(fm).squeeze(-1)
            return {'ct_hm': detect_outs['ct_hm'], 'pred_ts': pred_wts, 'semantic_mask': self.semantic_conv(outs)}
        else:
            coeffs_pred, detection = decode_ct_hm(torch.sigmoid(detect_outs['ct_hm']), detect_outs['coeff'])
            # 0.05 is the min_ct_score
            valid = detection[0, :, 2] >= 0.05
            if not valid.any():
                return None
            coeffs, detection = coeffs_pred[0][valid], detection[0][valid]
            coeffs = coeffs.unsqueeze(-1).unsqueeze(-1)
            fm = (proto_out * coeffs).sum(dim=1)
            pred_wts = self.predict(fm).squeeze(-1)
            return {'detection': detection, 'pred_ts': pred_wts}