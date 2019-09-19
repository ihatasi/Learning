import chainer, os, argparse
from chainer.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cupy
from chainercv.datasets import TransformDataset
from chainercv.transforms import resize


def main():
    parser = argparse.ArgumentParser(description="Vanilla_AE")
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--snapshot", "-s", type=int, default=10)
    parser.add_argument("--n_dimz", "-z", type=int, default=16)
    parser.add_argument("--network", "-n", type=str, default='conv')

    args = parser.parse_args()
    if args.network=='conv':
        import Network.mnist_conv as Network
    elif args.network=='fl':
        import Network.mnist_fl as Network
    else:
        raise Exception('Error!')
    out = os.path.join('Pred', args.network, 'epoch_{}'.format(args.snapshot))
    os.makedirs(out, exist_ok=True)

    def transform(in_data):
        img, label = in_data
        img = resize(img, (32, 32))
        return img, label

    def plot_mnist_data(samples, label1, label2):
        for index, data in enumerate(samples):#(配列番号，要素)
            plt.subplot(5, 13, index + 1)#(行数, 列数, 何番目のプロットか)
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
        pict = os.path.join(out, "{}_{}.png".format(label1, label2))
        plt.savefig(pict)
        plt.show()

    batchsize = args.batchsize
    gpu_id = args.gpu
    _, test = mnist.get_mnist(withlabel=True, ndim=3)
    test = TransformDataset(test, transform)
    xp = cupy
    AE = Network.AE(n_dimz=args.n_dimz, batchsize=args.batchsize)
    Critic = Network.Critic()
    
    AE.to_gpu()
    Critic.to_gpu()
    AE_path = os.path.join('result', args.network, 'AE_snapshot_epoch_{}.npz'.format(args.snapshot))
    Critic_path = os.path.join('result', args.network, 'Critic_snapshot_epoch_{}.npz'.format(args.snapshot))
    chainer.serializers.load_npz(AE_path, AE)
    chainer.serializers.load_npz(Critic_path, Critic)
    label1 = 1
    label2 = 9
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
        in1 = (data1*255).astype(np.uint8).reshape(32, 32)
        in2 = (data2*255).astype(np.uint8).reshape(32, 32)
        z_diff = (z2 - z1).data
        z_itp = z1.data#start point

        itp_list.append(in1)
        for j in range(1, 11):#間の座標を出す
            z_itp = xp.vstack((z_itp, z1.data+z_diff/10*j))
        for k in range(0,11):#端から座標を移動して画像を出力
            itp = AE(xp.copy(z_itp[k][None, ...]), z_itp[k], train=False)
            itp_out = (itp.data*255).astype(np.uint8).reshape(32, 32)
            itp_list.append(chainer.cuda.to_cpu(itp_out))
        itp_list.append(in2)

    plot_mnist_data(itp_list, label1, label2)


if __name__ == '__main__':
    main()
