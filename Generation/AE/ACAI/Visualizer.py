import os
import chainer
import numpy as np
import chainer.backends.cuda
from PIL import Image
from chainer import Variable
import matplotlib.pyplot as plt
import cupy

def out_generated_image(Enc, Dec, Critic, test1, test2, out):
    @chainer.training.make_extension()
    def plot_mnist_data(samples, trainer, out):
        pict = os.path.join(out, 'pict')
        os.makedirs(pict, exist_ok=True)
        pict = os.path.join(pict, 'epoch_{}'.format(trainer.updater.epoch))
        for index, data in enumerate(samples):#(配列番号，要素)
            plt.subplot(1, 13, index + 1)#(行数, 列数, 何番目のプロットか)
            plt.axis('off')#軸はoff
            plt.imshow(data.reshape(32, 32), cmap="gray")#nearestで補完
            if index == 0:
                plt.title("in_1", color='red')
            elif index == 12:
                plt.title("in_2", color='red')
            elif index == 1:
                plt.title("out_1", color='red')
            elif index == 11:
                plt.title("out_2", color='red')
        plt.savefig(pict)
        plt.clf()
    def make_image(trainer):
        xp = cupy
        itp_list = []
        for i in range(0,1):
            data1 = test1[i]
            data2 = test2[i]
            with chainer.using_config('train', False):
                z1 = Enc(xp.array([data1]).astype(np.float32))
                z2 = Enc(xp.array([data2]).astype(np.float32))
                y1 = Dec(z1)
                y2 = Dec(z2)
            in1 = (data1*255).astype(np.uint8).reshape(32, 32)
            in2 = (data2*255).astype(np.uint8).reshape(32, 32)
            z_diff = (z2 - z1).data
            z_itp = z1.data#start point

            itp_list.append(in1)
            for j in range(1, 11):#間の座標を出す
                z_itp = xp.vstack((z_itp, z1.data+z_diff/10*j))
            for k in range(0,11):#端から座標を移動して画像を出力
                with chainer.using_config('train', False):
                    itp = Dec(xp.copy(z_itp[k][None, ...]))
                    itp = chainer.backends.cuda.to_cpu(itp.data)
                itp_out = (itp*255).astype(np.uint8).reshape(32, 32)
                itp_list.append(itp_out)
            itp_list.append(in2)

        plot_mnist_data(itp_list, trainer, out)

    return make_image
