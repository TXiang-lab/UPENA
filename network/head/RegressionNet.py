import torch.nn as nn
import torch


class RegressionNet(nn.Module):
    def __init__(self, planes, mask_dim):
        super(RegressionNet, self).__init__()

        # self.conv = nn.Sequential(
        #     nn.Conv2d(planes, planes, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        # self.ct_hm = nn.Conv2d(planes, 2, kernel_size=3, stride=1, padding=1, bias=True)
        # self.coeff = nn.Conv2d(planes, mask_dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.ct_hm = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.ct_hm[-1].bias.data.fill_(-2.19)

        self.coeff = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, mask_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )


    def forward(self, x):
        return {'ct_hm':self.ct_hm(x), 'coeff':self.coeff(x)}



def coeff_net(channels, mask_dim):
    return RegressionNet(channels, mask_dim)
