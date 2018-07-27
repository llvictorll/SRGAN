import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import torch.nn.functional as F


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, imgFolder,transform=transforms.ToTensor()):
        super(CelebADataset, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.list = os.listdir(self.imgFolder)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        file = os.path.join(self.imgFolder,self.list[i])
        image = Image.open(file).crop((15,15,175,175))
        imgx = self.transform(image)
        if imgx.size(0) == 1:
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
        imgxb = F.avg_pool2d(imgx, 8, stride=4, padding=2)
        return imgx, imgxb