# some base nn layers copied from the code of the 1st place team of the 3rd Kaggle Data Science Bowl competition.

import torch
from torch import nn


class PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class Rec3(nn.Module):
    def __init__(self, n0, n1, n2, n3, p=0.0, integrate=True):
        super(Rec3, self).__init__()

        self.block01 = nn.Sequential(
            nn.Conv3d(n0, n1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(n1),
            nn.ReLU(),
            nn.Conv3d(n1, n1, kernel_size=3, padding=1),
            nn.BatchNorm3d(n1))

        self.block11 = nn.Sequential(
            nn.Conv3d(n1, n1, kernel_size=3, padding=1),
            nn.BatchNorm3d(n1),
            nn.ReLU(),
            nn.Conv3d(n1, n1, kernel_size=3, padding=1),
            nn.BatchNorm3d(n1))

        self.block21 = nn.Sequential(
            nn.ConvTranspose3d(n2, n1, kernel_size=2, stride=2),
            nn.BatchNorm3d(n1),
            nn.ReLU(),
            nn.Conv3d(n1, n1, kernel_size=3, padding=1),
            nn.BatchNorm3d(n1))

        self.block12 = nn.Sequential(
            nn.Conv3d(n1, n2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(n2),
            nn.ReLU(),
            nn.Conv3d(n2, n2, kernel_size=3, padding=1),
            nn.BatchNorm3d(n2))

        self.block22 = nn.Sequential(
            nn.Conv3d(n2, n2, kernel_size=3, padding=1),
            nn.BatchNorm3d(n2),
            nn.ReLU(),
            nn.Conv3d(n2, n2, kernel_size=3, padding=1),
            nn.BatchNorm3d(n2))

        self.block32 = nn.Sequential(
            nn.ConvTranspose3d(n3, n2, kernel_size=2, stride=2),
            nn.BatchNorm3d(n2),
            nn.ReLU(),
            nn.Conv3d(n2, n2, kernel_size=3, padding=1),
            nn.BatchNorm3d(n2))

        self.block23 = nn.Sequential(
            nn.Conv3d(n2, n3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(n3),
            nn.ReLU(),
            nn.Conv3d(n3, n3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3))

        self.block33 = nn.Sequential(
            nn.Conv3d(n3, n3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3),
            nn.ReLU(),
            nn.Conv3d(n3, n3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3))

        self.relu = nn.ReLU()
        self.p = p
        self.integrate = integrate

    def forward(self, x0, x1, x2, x3):
        if self.p > 0 and self.training:
            coef = torch.bernoulli((1.0 - self.p) * torch.ones(8))
            out1 = coef[0] * self.block01(x0) + coef[1] * self.block11(x1) + coef[2] * self.block21(x2)
            out2 = coef[3] * self.block12(x1) + coef[4] * self.block22(x2) + coef[5] * self.block32(x3)
            out3 = coef[6] * self.block23(x2) + coef[7] * self.block33(x3)
        else:
            out1 = (1 - self.p) * (self.block01(x0) + self.block11(x1) + self.block21(x2))
            out2 = (1 - self.p) * (self.block12(x1) + self.block22(x2) + self.block32(x3))
            out3 = (1 - self.p) * (self.block23(x2) + self.block33(x3))

        if self.integrate:
            out1 += x1
            out2 += x2
            out3 += x3

        return x0, self.relu(out1), self.relu(out2), self.relu(out3)
