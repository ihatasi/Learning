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

    #import .py
    import Updater
    import Visualize
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
    dis = Network.Discriminator()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
    #Make optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = optimizers.Adam(alpha=alpha, beta1=beta1) #init_lr = alpha
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    #Get dataset
    train, _ = mnist.get_mnist(withlabel=True, ndim=3)
    train = [i[0] for i in train if(i[1]==1)] #ラベル1のみを選択
    #Setup iterator
    train_iter = iterators.SerialIterator(train, args.batchsize)
    #Setup updater
    updater = Updater.DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={'gen':opt_gen, 'dis':opt_dis},
        device=args.gpu)

    #Setup trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)
    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(
        trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'gen/loss', 'dis/loss', 'elapsed_time'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(Visualize.out_generated_image(
        gen, dis,
        10, 10, args.seed, args.out, args.dataset),
        trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
