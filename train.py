import torch.optim as optim
from network import *
import torchvision.transforms as transforms
from utils import *

from sacred import Experiment
from sacred.observers import MongoObserver

from tqdm import tqdm
from dataset import CelebADataset

ex = Experiment('test')
#ex.observers.append(MongoObserver.create(url='mongodb://besnier:XXXXX@drunk',db_name='VictorSacred'))

@ex.config
def conf():
    device = 'cuda:0'
    netG = NetG().cuda()
    netD = NetD().cuda()
    optimizerG = optim.Adam(netG.parameters(), 0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), 0.0001, betas=(0.5, 0.999))
    epoch = 100
    cuda = True
    param = None
    file = 'SRGAN'

    dataset = CelebADataset("/net/girlschool/besnier/CelebA_dataset/img_align_celeba",
                            transforms.Compose([transforms.Resize(128),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)


@ex.automain
def main(netG, netD, epoch, cuda, dataloader, optimizerG, optimizerD, file):
    netG.train()
    netD.train()
    sauvegarde_init(file)
    cpt = 0
    dTrue = []
    dFalse = []
    mse = []
    ref = []

    turn = True
    bar_epoch = tqdm(range(epoch))
    for e in bar_epoch:
        for i, (xhq, xlq) in zip(tqdm(range(len(dataloader))), dataloader):
            if turn:
                save_xb = xlq
                print_img(save_xb, 'image_de_base_bruit', file)
                print_img(xhq, 'image_de_base_sans_bruit', file)
                turn = False
                if cuda:
                    save_xb = save_xb.cuda()

            real_label = torch.FloatTensor(xlq.size(0)).fill_(.9)
            fake_label = torch.FloatTensor(xlq.size(0)).fill_(.1)

            if cuda:
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()
                xlq = xlq.cuda()
                xhq = xhq.cuda()

            # train D
            optimizerD.zero_grad()

            # avec de vrais labels
            outputTrue = netD(xhq)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            # avec de faux labels
            fake = netG(xlq)
            outputFalse = netD(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerD.step()

            # train G

            optimizerG.zero_grad()
            outputG = netG(xlq)
            outputDG = netD(outputG)
            lossGAN = F.binary_cross_entropy_with_logits(outputDG, real_label)
            lossMSE = F.mse_loss(outputG, xhq)

            (lossGAN*0.1+lossMSE).backward()
            optimizerG.step()

            #test
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())
            mse.append(F.mse_loss(netG(xlq).detach(), xhq))
            ref.append(F.mse_loss(F.upsample(xlq, scale_factor=4), xhq))

            bar_epoch.set_postfix({"Dataset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean(),
                                   "qualité": np.array(mse).mean(),
                                   "ref": np.array(ref).mean()})

            sauvegarde(file, np.array(dTrue).mean(), np.array(dFalse).mean(), np.array(mse).mean(), np.array(ref).mean())

            if i % 250 == 0:
                printG(save_xb, cpt, netG, file)
                cpt += 1
                dTrue = []
                dFalse = []
                mse = []
                ref = []
        if e % 2 == 1:
            torch.save({
                "generator":
                    {
                        'epoch': e + 1,
                        'state_dict': netG.state_dict(),
                        'optimizer': optimizerG.state_dict(),
                    },
                "discriminator":
                    {
                        'epoch': e + 1,
                        'state_dict': netD.state_dict(),
                        'optimizer': optimizerD.state_dict(),
                    }
            }, "/net/girlschool/besnier/model/model_srgan.pytorch")
