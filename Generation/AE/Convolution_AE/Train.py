#module
import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from chainer import optimizers, Variable, training
from chainer import iterators, datasets, serializers
from chainer.datasets import mnist, split_dataset_random
from chainer.training import extensions

#train
def main():
    parser = argparse.ArgumentParser(description="Vanilla_AE")
    parser.add_argument("--batchsize", "-b", type=int, default=64)
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--snapshot", "-s", type=int, default=10)
    parser.add_argument("--n_dimz", "-z", type=int, default=64)
    parser.add_argument("--dataset", "-d", type=str, default='mnist')

    args = parser.parse_args()
    #import program
    import Updater
    import Evaluator

    #print settings
    print("GPU:{}".format(args.gpu))
    print("epoch:{}".format(args.epoch))
    print("Minibatch_size:{}".format(args.batchsize))
    print('')

    batchsize = args.batchsize
    gpu_id = args.gpu
    max_epoch = args.epoch

    if args.dataset == "mnist":
        train_val, test = mnist.get_mnist(withlabel=False, ndim=3,
            scale=255.)
        import Network.mnist_net as Network
    else:
        train_val, test = chainer.datasets.get_cifar10(withlabel=False,
             scale=255.)
        import Network.cifar10_net as Network
    train, valid = split_dataset_random(train_val, 50000, seed=0)

    model = Network.AE(n_dimz=args.n_dimz)

    #set iterator
    train_iter = iterators.SerialIterator(train, batchsize)
    valid_iter = iterators.SerialIterator(valid, batchsize,
        repeat=False, shuffle=False)
    #optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        return optimizer
    opt = make_optimizer(model)
    #trainer
    updater = Updater.AEUpdater(
    model=model,
    iterator=train_iter,
    optimizer=opt,
    device=args.gpu)


    trainer = training.Trainer(updater, (max_epoch, 'epoch'),
     out='result')
    #trainer.extend(extensions.ExponentialShift('lr', 0.5),trigger=(30, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='log'))
    trainer.extend(Evaluator.AEEvaluator(
        iterator=valid_iter,
        target=model,
        device=args.gpu))
    trainer.extend(extensions.snapshot_object(model,
        filename='model_snapshot_epoch_{.updater.epoch}.npz'), trigger=(args.snapshot, 'epoch'))
    #trainer.extend(extensions.snapshot_object(optimizer, filename='optimizer_snapshot_epoch_{.updater.epoch}'), trigger=(args.snapshot, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss',
        'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    del trainer

if __name__ == '__main__':
    main()
