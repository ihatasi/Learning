#!/usr/bin/python3
import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from chainer import Variable


parser = argparse.ArgumentParser(description="DCGAN")
parser.add_argument("--n_dimz", "-z", type=int, default=128)
parser.add_argument("--dataset", "-ds", type=str, default="mnist")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

if args.dataset == "cifar10":
    import Network.cifar10_net as Network
else:
    import Network.mnist_net as Network


gen = Network.Generator(n_hidden=args.n_dimz)
gen.to_cpu()
load_path = 'result/{}/gen_epoch_100.npz'.format(args.dataset)
chainer.serializers.load_npz(load_path, gen)

np.random.seed(args.seed)
dtype = chainer.get_dtype()
hidden = np.random.uniform(-1, 1, (1, args.n_dimz, 1, 1)).astype(dtype)
z = Variable(np.asarray(hidden))
with chainer.using_config('train', False):
    x = gen(z)
np.random.seed()
x = np.asarray(np.clip(x.data * 255, 0.0, 255.0), dtype=np.uint8)
_, _, H, W = x.shape
if args.dataset == "mnist":
    x = x.reshape((1, 1, 1, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((1 * H, 1 * W))
else:
    x = x.reshape((1, 1, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((1 * H, 1 * W, 3))
preview_dir = './'
preview_path = preview_dir +\
    'test.png'
if not os.path.exists(preview_dir):
    os.makedirs(preview_dir)
Image.fromarray(x).save(preview_path)
plt.imshow(x)
plt.show()
