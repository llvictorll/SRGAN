import scipy.misc
import torchvision.utils as vutils
import numpy as np
import os


def imshow(img):
    img = img / 2 + 0.5
    npimg = (img.cpu()).numpy()
    return np.transpose(npimg, (1, 2, 0))


def printG(x, k, netG,file):
    o = netG(x)
    scipy.misc.imsave('/local/besnier/'+file+'/g{}.png'.format(k), imshow(vutils.make_grid(o.data)))


def print_img(x, name, file):
    scipy.misc.imsave('/local/besnier/'+file +'/'+ name + '.png', imshow(vutils.make_grid(x).data))


def sauvegarde_init(file):
    if not os.path.exists("/local/besnier/"+file):
        os.makedirs("/local/besnier/"+file)
    with open("/local/besnier/"+file+"/res.csv", 'a') as f:
        f.write('dTrue' + '\t' + 'dFalse' + '\t' + 'qualité_test' + '\t' + 'qualité_train' + '\t' + 'référence' + '\n')


def sauvegarde(file, *agr):
    with open("/local/besnier/"+file+"/res.csv", 'a') as f:
        for a in agr:
            f.write(str(a) + '\t')
        f.write('\n')