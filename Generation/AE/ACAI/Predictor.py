import chainer, os, argparse
from chainer.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cupy

import Network.mnist_net as Network

def main():
    parser = argparse.ArgumentParser(description="Vanilla_AE")
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--snapshot", "-s", type=int, default=10)
    parser.add_argument("--n_dimz", "-z", type=int, default=16)

    args = parser.parse_args()
    os.makedirs('Pred', exist_ok=True)

    def plot_mnist_data(samples, label1, label2):
        for index, data in enumerate(samples):#(配列番号，要素)
            plt.subplot(5, 13, index + 1)#(行数, 列数, 何番目のプロットか)
            plt.axis('off')#軸はoff
            plt.imshow(data.reshape(28, 28), cmap="gray")#nearestで補完
            if index == 0:
                plt.title("in_1", color='red')
            elif index == 12:
                plt.title("in_2", color='red')
            elif index == 1:
                plt.title("out_1", color='red')
            elif index == 11:
                plt.title("out_2", color='red')
        plt.savefig("Pred/{}_{}_epoch_{}.png".format(label1, label2, args.snapshot))
        plt.show()

    batchsize = args.batchsize
    gpu_id = args.gpu
    _, test = mnist.get_mnist(withlabel=True, ndim=3)
    xp = cupy
    AE = Network.AE(n_dimz=args.n_dimz)
    Critic = Network.Critic()
    
    AE.to_gpu()
    Critic.to_gpu()
    load_AE = 'result/AE_snapshot_epoch_{}.npz'.format(args.snapshot)
    load_Critic = 'result/Critic_snapshot_epoch_{}.npz'.format(args.snapshot)
    chainer.serializers.load_npz(load_AE, AE)
    chainer.serializers.load_npz(load_Critic, Critic)
    label1 = 1
    label2 = 0
    test1 = [i[0] for i in test if(i[1]==label1)]
    test2 = [i[0] for i in test if(i[1]==label2)]
    test1 = test1[0:5]
    test2 = test2[5:10]
    itp_list = []
    for i in range(0,5):
        data1 = test1[i]
        data2 = test2[i]
        y1, y2, yc, alpha, z1, z2 = AE(xp.array([data1]).astype(np.float32),
            xp.array([data2]).astype(np.float32))
        in1 = (data1*255).astype(np.uint8).reshape(28, 28)
        in2 = (data2*255).astype(np.uint8).reshape(28, 28)
        z_diff = (z2 - z1).data
        z_itp = z1.data#start point

        itp_list.append(in1)
        for j in range(1, 11):#間の座標を出す
            z_itp = xp.vstack((z_itp, z1.data+z_diff/10*j))
        for k in range(0,11):#端から座標を移動して画像を出力
            itp = AE(xp.copy(z_itp[k][None, ...]), z_itp[k], train=False)
            itp_out = (itp.data*255).astype(np.uint8).reshape(28, 28)
            itp_list.append(chainer.cuda.to_cpu(itp_out))
        itp_list.append(in2)

    plot_mnist_data(itp_list, label1, label2)


if __name__ == '__main__':
    main()
