
# detector net copied from the code of the 1st place team of the 3rd Kaggle Data Science Bowl competition.

import torch
from torch import nn

from .base_layer import PostRes


class ResNoduleNet(nn.Module):
    def __init__(self, num_anchors=3):
        super(ResNoduleNet, self).__init__()
        self.num_anchors = num_anchors
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3  # zhw coords
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.drop = nn.Dropout3d(p=0.5)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv3d(64, 5 * self.num_anchors, kernel_size=1))

    def forward(self, x, coord, output_feature=False):
        # x:     bs, 1, slice, height, width
        # coord: bs, 3, slice/4, height/4, width/4
        out = self.preBlock(x)
        # out:       bs, 24, slice, height, width
        out_pool = self.maxpool1(out)
        # out_pool:  bs, 24, slice/2, height/2, width/2
        out1 = self.forw1(out_pool)
        # out1:      bs, 32, slice/2, height/2, width/2
        out1_pool = self.maxpool2(out1)
        # out1_pool: bs, 32, slice/4, height/4, width/4
        out2 = self.forw2(out1_pool)
        # out2:      bs, 64, slice/4, height/4, width/4
        out2_pool = self.maxpool3(out2)
        # out2_pool: bs, 64, slice/8, height/8, width/8
        out3 = self.forw3(out2_pool)
        # out3:      bs, 64, slice/8, height/8, width/8
        out3_pool = self.maxpool4(out3)
        # out3_pool: bs, 64, slice/16, height/16, width/16
        out4 = self.forw4(out3_pool)
        # out4:      bs, 64, slice/16, height/16, width/16

        rev3 = self.path1(out4)
        # rev3:      bs, 64, slice/8, height/8, width/8
        comb3 = self.back3(torch.cat((rev3, out3), 1))  # 64 + 64
        # comb3:     bs, 64, slice/8, height/8, width/8
        rev2 = self.path2(comb3)
        # rev2:      bs, 64, slice/4, height/4, width/4

        comb2 = self.back2(torch.cat((rev2, out2, coord), 1))  # 64+64+3
        # comb2:     bs, 128, slice/4, height/4, width/4
        comb2_drop = self.drop(comb2)

        out = self.output(comb2_drop)
        # out:       bs, 15, slice/4, height/4, width/4
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        # out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], self.num_anchors, 5)
        # out:       bs, slice/4, height/4, width/4, 3, 5

        if output_feature:
            return out, comb2
        else:
            return out
