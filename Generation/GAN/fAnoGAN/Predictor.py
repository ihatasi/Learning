import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
import chainer.functions as F
from PIL import Image
from chainer import Variable
from chainer.datasets import mnist

parser = argparse.ArgumentParser(description="DCGAN")
parser.add_argument("--n_dimz", "-z", type=int, default=128)
parser.add_argument("--dataset", "-ds", type=str, default="mnist")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--PreNet", "-pn", type=str, default="WGANgp")
parser.add_argument("--method", "-m", type=str, default="ziz")
parser.add_argument("--epoch", "-e", type=int, default=500)
parser.add_argument("--Premodel", "-pm", type=int, default=500)    
args = parser.parse_args()

import Network.mnist_net as Network
#Get Pretrain Net
if args.PreNet == "WGANgp":
    import WGANgp.Network.mnist_net as PreNetwork
else:
    import WGAN.Network.mnist_net as PreNetwork

train_valid, test = mnist.get_mnist(withlabel=True, ndim=3)
test = [i[0] for i in test  if(i[1]==1)] #ラベル1のみを選択
test = test[0]

gen = PreNetwork.DCGANGenerator(n_hidden=args.n_dimz)
dis = PreNetwork.WGANDiscriminator()
enc = Network.AE()

gen.to_cpu()
enc.to_cpu()
dis.to_cpu()
load_path = 'result/{}/gen_epoch_{}.npz'.format(args.method, args.Premodel)
chainer.serializers.load_npz(load_path, gen)
load_path = 'result/{}/enc_epoch_{}.npz'.format(args.method, args.epoch)
chainer.serializers.load_npz(load_path, enc)
load_path = 'result/{}/dis_epoch_{}.npz'.format(args.method, args.Premodel)
chainer.serializers.load_npz(load_path, dis)
x_real = np.reshape(test[0],(1, 1, 28, 28))
k=1
ch = 512
with chainer.using_config('train', False):
    enc_z = enc(x_real)
    rec_x = gen(enc_z)
    
    _,f_x = dis(x_real)
    _,f_rec_x = dis(rec_x)

    L1 = F.mean_squared_error(x_real, rec_x)
    L2 = F.mean_squared_error(f_x, f_rec_x)/ch
    if args.method == "izif":
        A_score = L1.data + k*L2.data
    else:
        A_score = L1.data
rec_x = np.asarray(np.clip(rec_x.data * 255, 0.0, 255.0), dtype=np.uint8)
rec_x = rec_x.reshape(28, 28)
x_real = x_real.reshape(28, 28)*255
imgs = np.concatenate((x_real, rec_x), axis=1)

preview_dir = './'
preview_path = preview_dir +\
    '{}_pred.png'.format(args.method)
if not os.path.exists(preview_dir):
    os.makedirs(preview_dir)
plt.imshow(imgs, cmap='gray')
plt.title('A_score='+str(A_score), color='black')
plt.savefig(preview_path)