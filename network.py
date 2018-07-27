import torch
import torch.nn as nn
import torch.nn.functional as F


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.convInit = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.convRes1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.convResInit = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.convOut = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.pix_shuf1 = nn.PixelShuffle(2)
        self.pix_shuf2 = nn.PixelShuffle(2)
        # self.pix_shuf3 = nn.PixelShuffle(2)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.prelu6 = nn.PReLU()
        # self.prelu7 = nn.PReLU()

        self.batch = nn.BatchNorm2d(64, affine=False)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.prelu1(self.convInit(x))
        xres1 = self.batch(self.convRes2(self.prelu2(self.batch(self.convRes1(x)))))+x
        xres2 = self.batch(self.convRes3(self.prelu3(self.batch(self.convRes3(xres1)))))+xres1
        xres3 = self.batch(self.convRes5(self.prelu4(self.batch(self.convRes4(xres2)))))+xres2

        x = self.batch(self.convResInit(xres3))+x

        x = self.prelu5(self.pix_shuf1(self.conv1(x)))
        x = self.prelu6(self.pix_shuf2(self.conv2(x)))
        # x = self.prelu7(self.pix_shuf3(self.conv3(x)))

        x = self.tanh(self.convOut(x))
        return x


class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=2, padding=0, bias=False)
        self.batch1 = nn.BatchNorm2d(128, affine=False)
        self.batch2 = nn.BatchNorm2d(256, affine=False)
        self.batch3 = nn.BatchNorm2d(512, affine=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.batch1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.batch2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.batch3(self.conv4(x)), 0.2)
        x = self.conv5(x)
        return x.squeeze()
