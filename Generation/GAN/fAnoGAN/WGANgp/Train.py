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
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--snapshot_interval", "-s", type=int, default=50)
    parser.add_argument("--display_interval", "-d", type=int, default=1)
    parser.add_argument("--n_dimz", "-z", type=int, default=128)
    parser.add_argument("--dataset", "-ds", type=str, default="mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", "-o", type=str, default="result")
    parser.add_argument("--resume", '-r', default='')
    args = parser.parse_args()

    #import .py
    import Updater
    import Visualize
    import Network.mnist_net as Network

    #print settings
    print("GPU:{}".format(args.gpu))
    print("max_epoch:{}".format(args.epoch))
    print("Minibatch_size:{}".format(args.batchsize))
    print("Dataset:{}".format(args.dataset))
    print('')
    out = os.path.join(args.out, args.dataset)
    #Set up NN
    gen = Network.DCGANGenerator()
    dis = Network.WGANDiscriminator()
    
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    #Make optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.0, beta2=0.9):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    #Get dataset
    train, _ = mnist.get_mnist(withlabel=True, ndim=3, scale=1.)
    train = [i[0] for i in train if(i[1]==1)] #ラベル1のみを選択

    #Setup iterator
    train_iter = iterators.SerialIterator(train, args.batchsize)
    #Setup updater
    updater = Updater.WGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={'gen':opt_gen, 'dis':opt_dis},
        n_dis=5,
        lam=10,
        device=args.gpu)

    #Setup trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)
    snapshot_interval = (args.epoch, 'epoch')
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
    trainer.extend(extensions.LogReport(
        trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss', 'loss_grad', 'wasserstein_distance', 'elapsed_time'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(Visualize.out_generated_image(
        gen, dis,
        10, 10, args.seed, args.out, args.dataset),
        trigger=display_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
