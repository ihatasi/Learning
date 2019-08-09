#!/usr/bin/python3
import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from chainer import optimizers, Variable, training
from chainer import iterators, datasets, serializers
from chainer.datasets import mnist, split_dataset_random
from chainer.training import extensions

def main():
    parser = argparse.ArgumentParser(description="DCGAN")
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--snapshot_interval", "-s", type=int, default=10)
    parser.add_argument("--display_interval", "-d", type=int, default=1)
    parser.add_argument("--n_dimz", "-z", type=int, default=100)
    parser.add_argument("--dataset", "-ds", type=str, default="mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", "-o", type=str, default="result")
    parser.add_argument("--resume", '-r', default='')
    args = parser.parse_args()

class WeightClipping(object):
    name = 'WeightClipping'

    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, opt):
        for param in opt.target.params():
            xp = chainer.cuda.get_array_module(param.data)
            param.data = xp.clip(param.data, -self.threshold, self.threshold)

    #import .py
    import Updater
    import Visualize
    if args.dataset == "cifar10":
        import Network.cifar10_net as Network
    else:
        import Network.mnist_net as Network
    #print settings
    print("GPU:{}".format(args.gpu))
    print("epoch:{}".format(args.epoch))
    print("Minibatch_size:{}".format(args.batchsize))
    print("Dataset:{}".format(args.dataset))
    print('')
    out = os.path.join(args.out, args.dataset)
    #Set up NN
    gen = Network.Generator(n_hidden=args.n_dimz)
    cri = Network.Critic()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        cri.to_gpu()

    #Make optimizer
    opt_gen = chainer.optimizers.RMSprop(5e-5)
    opt_gen.setup(gen)
    opt_gen.add_hook(chainer.optimizer.GradientClipping(1))

    opt_cri = chainer.optimizers.RMSprop(5e-5)
    opt_cri.setup(gen)
    opt_cri.add_hook(chainer.optimizer.GradientClipping(1))
    opt_cri.add_hook(WeightClipping(0.01))

    #Get dataset
    if args.dataset == "mnist":
        train, _ = mnist.get_mnist(withlabel=False, ndim=3, scale=255.)
    else:
        train, _ = chainer.datasets.get_cifar10(withlabel=False, scale=255.)
    #Setup iterator
    train_iter = iterators.SerialIterator(train, args.batchsize)
    #Setup updater
    updater = Updater.WGANUpdater(
        models=(gen, cri),
        iterator=train_iter,
        optimizer={'gen':opt_gen, 'cri':opt_cri},
        n_c = 5,
        device=args.gpu)

    #Setup trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)
    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        cri, 'cri_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(
        trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'cri/loss', 'elapsed_time'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(Visualize.out_generated_image(
        gen, cri,
        10, 10, args.seed, args.out, args.dataset),
        trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
