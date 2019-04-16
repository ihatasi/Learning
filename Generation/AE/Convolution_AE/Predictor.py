import chainer, os, argparse
from chainer.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import Network

def main():
    parser = argparse.ArgumentParser(description="Vanilla_AE")
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--snapshot", "-s", type=int, default=100)
    parser.add_argument("--n_dimz", "-z", type=int, default=64)

    args = parser.parse_args()
    os.makedirs('pict', exist_ok=True)

    def plot_mnist_data(samples):
        for index, (data, label) in enumerate(samples):#(配列番号，要素)
            plt.subplot(5, 5, index + 1)#(行数, 列数, 何番目のプロットか)
            plt.axis('off')#軸はoff
            plt.imshow(data.reshape(28, 28), cmap="gray")#nearestで補完
            plt.title(int(label), color='red')
        plt.savefig("pict/epoch_{}.png".format(args.snapshot))
        plt.show()



    batchsize = args.batchsize
    gpu_id = args.gpu
    _, test = mnist.get_mnist(withlabel=True, ndim=1)
    model = Network.AE(n_dimz=args.n_dimz, n_out=784)
    model.to_cpu()
    load_path = 'result/model_snapshot_epoch_{}.npz'.format(args.snapshot)
    chainer.serializers.load_npz(load_path, model)
    test = test[:25]
    pred_list = []
    for (data, label) in test:
        pred_data = model(np.array([data]).astype(np.float32)).data
        pred_list.append((pred_data, label))#2つで1セットをたくさん作ってる

    plot_mnist_data(pred_list)


if __name__ == '__main__':
    main()
