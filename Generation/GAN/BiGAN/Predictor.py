#!/usr/bin/python3
import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
import chainer.functions as F
from PIL import Image
from chainer import Variable
from chainer.datasets import mnist

parser = argparse.ArgumentParser(description="DCGAN")
parser.add_argument("--n_dimz", "-z", type=int, default=100)
parser.add_argument("--dataset", "-ds", type=str, default="mnist")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import Network.mnist_net as Network
train_valid, test = mnist.get_mnist(withlabel=True, ndim=3, scale=255.)
test = [i[0] for i in test  if(i[1]==8)] #ラベル1のみを選択
test = test[0:1]


gen = Network.Generator(n_hidden=args.n_dimz)
enc = Network.Encoder(n_hidden=args.n_dimz)
dis = Network.Discriminator()

gen.to_cpu()
enc.to_cpu()
dis.to_cpu()
load_path = 'result/{}/gen_epoch_100.npz'.format(args.dataset)
chainer.serializers.load_npz(load_path, gen)
load_path = 'result/{}/enc_epoch_100.npz'.format(args.dataset)
chainer.serializers.load_npz(load_path, enc)
load_path = 'result/{}/dis_epoch_100.npz'.format(args.dataset)
chainer.serializers.load_npz(load_path, dis)

x_real = np.reshape(test[0],(1, 1, 28, 28))/255.
with chainer.using_config('train', False):
    enc_z = enc(x_real)
    y_rl, y_real = dis(x_real, enc_z)

    x_fake = gen(enc_z)
    y_fl, y_fake = dis(x_fake, enc_z)
    L1 = F.sum(F.absolute_error(x_real, x_fake))
    L2 = F.sum(F.absolute_error(y_rl, y_fl))
    A_score = (1-0.1)*L1.data + 0.1*L2.data
x_fake = np.asarray(np.clip(x_fake.data * 255, 0.0, 255.0), dtype=np.uint8)
x_fake = x_fake.reshape(28, 28)
x_real = x_real.reshape(28, 28)*255
imgs = np.concatenate((x_real, x_fake), axis=1)

preview_dir = './'
preview_path = preview_dir +\
    'pred.png'
if not os.path.exists(preview_dir):
    os.makedirs(preview_dir)
plt.imshow(imgs, cmap='gray')
plt.title('A_score='+str(A_score), color='black')
plt.savefig(preview_path)
