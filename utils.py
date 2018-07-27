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
    scipy.misc.imsave('/net/girlschool/besnier/'+file+'/g{}.png'.format(k), imshow(vutils.make_grid(o.data)))


def print_img(x, name, file):
    scipy.misc.imsave('/net/girlschool/besnier/'+file +'/'+ name + '.png', imshow(vutils.make_grid(x).data))


def sauvegarde_init(file):
    if not os.path.exists("/net/girlschool/besnier/"+file):
        os.mkdir("/net/girlschool/besnier/"+file)
    with open("/net/girlschool/besnier/"+file+"/res.csv", 'a') as f:
        f.write('dTrue' + '\t' + 'dFalse' + '\t' + 'qualité' + '\t' + 'référence' + '\n')


def sauvegarde(file, *agr):
    with open("/net/girlschool/besnier/"+file+"/res.csv", 'a') as f:
        for a in agr:
            f.write(str(a) + '\t')
        f.write('\n')