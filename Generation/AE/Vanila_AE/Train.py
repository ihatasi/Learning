#module
import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from chainer import optimizers, Variable, training
from chainer import iterators, datasets, serializers
from chainer.datasets import mnist, split_dataset_random
from chainer.training import extensions
#program
import Network
import Updater
import Evaluator

parser = argparse.ArgumentParser(description="Vanilla_AE")
parser.add_argument("--batchsize", "-b", type=int, default=128)
parser.add_argument("--epoch", "-e", type=int, default=100)
parser.add_argument("--gpu", "-g", type=int, default=0)
parser.add_argument("--snapshot", "-s", type=int, default=100)
parser.add_argument("--n_dimz", "-z", type=int, default=2)

args = parser.parse_args()

#print settings
print("GPU:{}".format(args.gpu))
print("epoch:{}".format(args.epoch))
print("Minibatch_size:{}".format(args.batchsize))
print('')

#check save model file
os.makedirs('model', exist_ok=True)

#train
def main(model, train, valid):
    batchsize = args.batchsize
    gpu_id = args.gpu
    max_epoch = args.epoch
    #set iterator
    train_iter = iterators.SerialIterator(train, batchsize)
    valid_iter = iterators.SerialIterator(valid, batchsize)
    #optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        return optimizer
    opt = make_optimizer(model)
    #trainer
    updater = Updater.AEUpteder(train_iter, opt, device=gpu_id)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'),
     out='result')
    #trainer.extend(extensions.ExponentialShift('lr', 0.5),trigger=(30, 'epoch'))
    trainer.extend(extensions.LogReport(log_name='log'))
    trainer.extend(extensions.snapshot_object(model, filename='model_snapshot_epoch_{.updater.epoch}'), trigger=(args.snapshot, 'epoch'))
    #trainer.extend(extensions.snapshot_object(optimizer, filename='optimizer_snapshot_epoch_{.updater.epoch}'), trigger=(args.snapshot, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'valid/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    del trainer

if __name__ == '__main__':
    #dataset
    train_val, test = mnist.get_mnist(withlabel=False, ndim=1)
    train, valid = split_dataset_random(train_val, 50000, seed=0)
    model = Network.AE(args.n_dimz, 784)
    main(model, train=train, valid = valid)
