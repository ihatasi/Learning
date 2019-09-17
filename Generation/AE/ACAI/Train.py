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
    parser.add_argument("--n_dimz", "-z", type=int, default=2)
    parser.add_argument("--dataset", "-d", type=str, default='mnist')

    args = parser.parse_args()
    #import program
    import Updater
    import Evaluator
    import Visualizer

    #print settings
    print("GPU:{}".format(args.gpu))
    print("epoch:{}".format(args.epoch))
    print("Minibatch_size:{}".format(args.batchsize))
    print('')

    batchsize = args.batchsize
    gpu_id = args.gpu
    max_epoch = args.epoch

    train_val, _ = mnist.get_mnist(withlabel=False, ndim=3)
    #for visualize
    _, test = mnist.get_mnist(withlabel=True, ndim=3)
    label1 = 1
    label2 = 5
    test1 = [i[0] for i in test if(i[1]==label1)]
    test2 = [i[0] for i in test if(i[1]==label2)]
    test1 = test1[0:5]
    test2 = test2[5:10]

    import Network.mnist_net as Network
    train, valid = split_dataset_random(train_val, 50000, seed=0)

    AE = Network.AE(n_dimz=args.n_dimz)
    Critic = Network.Critic()

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
    opt_AE = make_optimizer(AE)
    opt_Critic = make_optimizer(Critic) 
    #trainer
    updater = Updater.ACAIUpdater(
    model=(AE, Critic),
    iterator=train_iter,
    optimizer={'AE':opt_AE, 'Critic':opt_Critic},
    device=args.gpu)


    trainer = training.Trainer(updater, (max_epoch, 'epoch'),
     out='result')
    #trainer.extend(extensions.ExponentialShift('lr', 0.5),trigger=(30, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='log'))
    #trainer.extend(Evaluator.AEEvaluator(
    #    iterator=valid_iter,
    #    target=model,
    #    device=args.gpu))
    snapshot_interval = (args.snapshot, 'epoch')
    display_interval = (1, 'epoch')
    trainer.extend(extensions.snapshot_object(AE,
        filename='AE_snapshot_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(Critic,
        filename='Critic_snapshot_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    #trainer.extend(extensions.snapshot_object(optimizer, filename='optimizer_snapshot_epoch_{.updater.epoch}'), trigger=(args.snapshot, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'Critic_loss',
        'AE_loss', 'rec_loss']), trigger=display_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(Visualizer.out_generated_image(AE, Critic, test1, test2),
        trigger=(10, 'epoch'))
    trainer.run()
    del trainer

if __name__ == '__main__':
    main()
