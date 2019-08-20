#!/usr/bin/python3
import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from chainer import optimizers, Variable, training
from chainer import iterators, datasets, serializers
from chainer.datasets import mnist, split_dataset_random
from chainer.training import extensions

import Visualize

class WeightClipping(object):
    name = 'WeightClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        for param in opt.target.params():
            xp = chainer.cuda.get_array_module(param.data)
            param.data = xp.clip(param.data, -self.threshold, self.threshold)

def main():
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument("--snapshot_interval", "-s", type=int, default=50)
    parser.add_argument("--display_interval", "-d", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", "-ds", type=str, default="mnist")
    parser.add_argument("--n_dimz", "-z", type=int, default=128)
    args = parser.parse_args()
    
    out = os.path.join(args.out, args.dataset)
    # Networks
    if args.dataset == "mnist":
        import Network.mnist_net as Network
    else:
        import Network.cifar10_net as Network
    gen = Network.DCGANGenerator(n_hidden=args.n_dimz)
    dis = Network.WGANDiscriminator()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    # Optimizers
    opt_gen = chainer.optimizers.RMSprop(5e-5)
    opt_gen.setup(gen)
    opt_gen.add_hook(chainer.optimizer.GradientClipping(1))

    opt_dis = chainer.optimizers.RMSprop(5e-5)
    opt_dis.setup(dis)
    opt_dis.add_hook(chainer.optimizer.GradientClipping(1))
    opt_dis.add_hook(WeightClipping(0.01))

    # Dataset
    if args.dataset == "mnist":
        train, _ = mnist.get_mnist(withlabel=False, ndim=3, scale=1.)
    else:
        train, _ = chainer.datasets.get_cifar10(withlabel=False, scale=1.)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Trainer
    import Updater
    updater = Updater.WGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={'gen':opt_gen, 'dis':opt_dis},
        n_dis=5,
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)

    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'epoch')

    # Extensions
    trainer.extend(extensions.dump_graph('wasserstein distance'))
    trainer.extend(extensions.snapshot(
        'snapshot_epoch_{.updater.epoch}.npz'
    ), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PlotReport(['wasserstein distance'],
                              'epoch', file_name='distance.png'))
    trainer.extend(
        extensions.PlotReport(
            ['gen/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'wasserstein distance', 'gen/loss', 'elapsed_time']), 
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(Visualize.out_generated_image(
        gen, dis,
        10, 10, args.seed, args.out, args.dataset),
        trigger=snapshot_interval)
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run
    trainer.run()

if __name__ == '__main__':
    main()
