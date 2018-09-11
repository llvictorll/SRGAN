import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from math import *
from tqdm import tqdm
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

class CelebADataset2(torch.utils.data.Dataset):
    def __init__(self, imgFolder, f_bruit, transform=transforms.ToTensor()):
        super(CelebADataset2, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.list = os.listdir(self.imgFolder)
        self.f_bruit = f_bruit

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        file = os.path.join(self.imgFolder, self.list[i])
        image = Image.open(file).crop((15, 15, 175, 175))
        imgx = self.transform(image)
        if imgx.size(0) == 1:
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
        imgxb = self.f_bruit(imgx)
        return imgx, imgxb


class YoutubeFacesDataset(torch.utils.data.Dataset):
    def __init__(self, ImgFolder, f_bruit, inf, sup, transform=transforms.ToTensor()):
        super(YoutubeFacesDataset, self).__init__()
        self.ImgFolder = ImgFolder
        self.list_p = os.listdir(self.ImgFolder)  # Contient la liste des personnalités
        self.transform = transform
        self.f_bruit = f_bruit
        self.dir = []
        for k in tqdm(range(len(self.list_p))):
            L = os.path.join(self.ImgFolder, self.list_p[k])
            Llist = os.listdir(L)
            for j in range(len(Llist)):
                L2 = os.path.join(L, Llist[j])
                L2list = os.listdir(L2)
                for k in range(len(L2list)):
                    self.dir.append(os.path.join(L2, L2list[k]))
        if inf < sup:
            self.dir_final = self.dir[0:ceil(len(self.dir) * sup / 100)]
        else:
            self.dir_final = self.dir[ceil(len(self.dir) * inf / 100):]

    def __len__(self):
        return len(self.dir_final)

    def __getitem__(self, i):
        img = Image.open(self.dir[i - 1]).crop((5, 5, 250, 250))
        imgx = self.transform(img)
        if imgx.size(0) == 1:
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))

        imgxr = self.f_bruit(imgx)
        return imgx, imgxr