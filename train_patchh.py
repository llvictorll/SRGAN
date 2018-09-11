import torch.optim as optim
from network import *
import torchvision.transforms as transforms
from utils import *
from module_bruit import F_bruit, Patch_block, Sup_res1, Sup_res2
from sacred import Experiment
from sacred.observers import MongoObserver
from module_bruit import F_bruit, Patch_block, Sup_res1, Sup_res2
from tqdm import tqdm
from dataset import *

ex = Experiment('test')
#ex.observers.append(MongoObserver.create(url='mongodb://besnier:XXXXX@drunk',db_name='VictorSacred'))

@ex.config
def conf():
    device = 'cuda:0'
    netG = NetG().cuda()
    netD = NetD_patch().cuda()
    optimizerG = optim.Adam(netG.parameters(), 0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), 0.0004, betas=(0.5, 0.999))
    epoch = 100
    cuda = True
    f_bruit = Sup_res2
    param = None
    file = 'SRGAN/SRGAN_base'
    f = f_bruit(param)
    trainset = CelebADataset2("/net/girlschool/besnier/CelebA_dataset/train",
                             f,
                             transforms.Compose([transforms.Resize(64),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ]))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

    testset = CelebADataset2("/net/girlschool/besnier/CelebA_dataset/test",
                             f,
                             transforms.Compose([transforms.Resize(64),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ]))

    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1, drop_last=True)

    datasetYtrain = YoutubeFacesDataset("/net/girlschool/besnier/YoutubeFaces",
                                        f,
                                        0,
                                        80,
                                        transforms.Compose([transforms.Resize(64),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                            ]))
    testloaderY = torch.utils.data.DataLoader(datasetYtrain, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

@ex.automain
def main(netG, netD, epoch, cuda, trainloader, testloader, testloaderY, optimizerG, optimizerD, file):
    netG.train()
    netD.train()
    sauvegarde_init(file)
    cpt = 0
    dTrue = []
    dFalse = []
    mse_train = []
    turn = True
    bar_epoch = tqdm(range(epoch))
    for e in bar_epoch:
        for i, (xhq, xlq) in zip(tqdm(range(len(trainloader))), trainloader):

            real_label = torch.FloatTensor(xlq.size(0)*4*4).fill_(.9)
            fake_label = torch.FloatTensor(xlq.size(0)*4*4).fill_(.1)

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

            (0.01*lossGAN+lossMSE).backward()
            # lossMSE.backward()
            optimizerG.step()
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())
            mse_train.append(lossMSE.data.mean())
            bar_epoch.set_postfix({"Dataset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean(),
                                   "lossMSE": np.array(mse_train).mean()})
            #test
            if i % 250 == 0:
                testbar = tqdm(range(len(testloaderY)))
                mse_test = []
                ref = []
                for j, (xhqt, xlqt) in zip(testbar, testloader):
                    if j > 100:
                        break
                    if turn:
                        save_xb = xlqt
                        print_img(xhqt, 'image_de_base_sans_bruit', file)
                        print_img(F.upsample(xlqt, scale_factor=2), 'ref_upsampling', file)
                        turn = False
                        if cuda:
                            save_xb = save_xb.cuda()
                    if cuda:
                        xlqt = xlqt.cuda()
                        xhqt = xhqt.cuda()

                    output = netG(xlqt).detach()
                    mse_test.append(F.mse_loss(output, xhqt).data.mean())
                    ref.append(F.mse_loss(F.upsample(xlqt, scale_factor=2), xhqt).data.mean())

                    testbar.set_postfix({"qualité": np.array(mse_test).mean(),
                                         "ref": np.array(ref).mean()})

                sauvegarde(file, np.array(dTrue).mean(), np.array(dFalse).mean(),
                           np.array(mse_test).mean(), np.array(mse_train).mean(), np.array(ref).mean())

                printG(save_xb, cpt, netG, file)
                cpt += 1

                mse_train = []
                dTrue = []
                dFalse = []

    for g in optimizerD.param_groups:
        g['lr'] = g['lr']*0.99
    for g in optimizerG.param_groups:
        g['lr'] = g['lr'] * 0.99


