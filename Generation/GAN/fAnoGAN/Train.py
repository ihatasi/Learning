#!/usr/bin/python3
import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from chainer import optimizers, Variable, training
from chainer import iterators, datasets, serializers
from chainer.datasets import mnist, split_dataset_random
from chainer.training import extensions


def main():
    parser = argparse.ArgumentParser(description="WGAN-gp")
    parser.add_argument("--batchsize", "-b", type=int, default=64)
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--snapshot_interval", "-s", type=int, default=50)
    parser.add_argument("--display_interval", "-d", type=int, default=1)
    parser.add_argument("--n_dimz", "-z", type=int, default=128)
    parser.add_argument("--dataset", "-ds", type=str, default="mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", "-o", type=str, default="result")
    parser.add_argument("--method", "-m", type=str, default="ziz")
    parser.add_argument("--resume", '-r', default='')
    parser.add_argument("--PreNet", "-pn", type=str, default="WGANgp")
    parser.add_argument("--Premodel", "-pm", type=int, default=500)    
    args = parser.parse_args()

    #import .py
    import Updater
    import Visualize
    import Network.mnist_net as Network
    #Get Pretrain Net
    if args.PreNet == "WGANgp":
        import WGANgp.Network.mnist_net as PreNetwork
    else:
        import WGAN.Network.mnist_net as PreNetwork


    #print settings
    print("GPU:{}".format(args.gpu))
    print("max_epoch:{}".format(args.epoch))
    print("Minibatch_size:{}".format(args.batchsize))
    print("Dataset:{}".format(args.dataset))
    print("Method:{}".format(args.method))
    print('')
    out = os.path.join(args.out, args.method)

    #Set up NN
    gen = PreNetwork.DCGANGenerator(n_hidden=args.n_dimz)
    dis = PreNetwork.WGANDiscriminator()
    enc = Network.AE()

    #Load PreTrain model
    load_path = '{}/result/mnist/gen_epoch_{}.npz'.format(args.PreNet, args.Premodel)
    chainer.serializers.load_npz(load_path, gen)
    load_path = '{}/result/mnist/dis_epoch_{}.npz'.format(args.PreNet, args.Premodel)
    chainer.serializers.load_npz(load_path, dis)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
        enc.to_gpu()

    #Make optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.0, beta2=0.9):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)
    opt_enc = make_optimizer(enc)

    #Get dataset
    train, _ = mnist.get_mnist(withlabel=True, ndim=3, scale=1.)
    train = [i[0] for i in train if(i[1]==1)] #ラベル1のみを選択

    #Setup iterator
    train_iter = iterators.SerialIterator(train, args.batchsize)
    #Setup updater
    updater_args={
        "models":(gen, dis, enc),
        "iterator":train_iter,
        "optimizer":{'gen':opt_gen, 'dis':opt_dis, 'enc':opt_enc},
        "n_dimz":args.n_dimz,
        "device":args.gpu}
    if args.method=='izif':
        updater = Updater.izifUpdater(**updater_args)
    elif args.method=='izi':
        updater = Updater.iziUpdater(**updater_args)
    elif args.method=='ziz':
        updater = Updater.zizUpdater(**updater_args)
    else:
        raise NotImplementedError()

    #Setup trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)
    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(
        trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'elapsed_time'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(Visualize.out_generated_image(
        gen, enc,
        10, 10, args.seed, out, args.dataset),
        trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
