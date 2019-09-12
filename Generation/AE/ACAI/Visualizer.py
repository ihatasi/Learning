import os
import chainer
import numpy as np
import chainer.backends.cuda
from PIL import Image
from chainer import Variable
import matplotlib.pyplot as plt
import cupy

def out_generated_image(AE, Critic, test1, test2):
    @chainer.training.make_extension()
    def plot_mnist_data(samples, trainer):
        os.makedirs('pict', exist_ok=True)
        for index, data in enumerate(samples):#(配列番号，要素)
            plt.subplot(1, 13, index + 1)#(行数, 列数, 何番目のプロットか)
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
        plt.savefig("pict/epoch_{}.png".format(trainer.updater.epoch))
        plt.clf()
    def make_image(trainer):
        xp = cupy
        itp_list = []
        for i in range(0,1):
            data1 = test1[i]
            data2 = test2[i]
            with chainer.using_config('train', False):
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
                with chainer.using_config('train', False):
                    itp = AE(z_itp[k], z_itp[k], train=False)
                    itp = chainer.backends.cuda.to_cpu(itp.data)
                itp_out = (itp*255).astype(np.uint8).reshape(28, 28)
                itp_list.append(itp_out)
            itp_list.append(in2)

        plot_mnist_data(itp_list, trainer)

    return make_image
