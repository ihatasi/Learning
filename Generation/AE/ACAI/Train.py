#module
import chainer, argparse, os
import numpy as np
import matplotlib.pyplot as plt
from chainer import optimizers, Variable, training
from chainer import iterators, datasets, serializers
from chainer.datasets import mnist, split_dataset_random
from chainer.training import extensions
from chainercv.datasets import TransformDataset
from chainercv.transforms import resize

#train
def main():
    parser = argparse.ArgumentParser(description="Vanilla_AE")
    parser.add_argument("--batchsize", "-b", type=int, default=64)
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--snapshot", "-s", type=int, default=10)
    parser.add_argument("--n_dimz", "-z", type=int, default=16)
    parser.add_argument("--dataset", "-d", type=str, default='mnist')
    parser.add_argument("--network", "-n", type=str, default='conv')

    args = parser.parse_args()

    def transform(in_data):
        img = in_data
        img = resize(img, (32, 32))
        return img
    def transform2(in_data):
        img, label = in_data
        img = resize(img, (32, 32))
        return img, label

    #import program
    import Updater
    import Visualizer

    #print settings
    print("GPU:{}".format(args.gpu))
    print("epoch:{}".format(args.epoch))
    print("Minibatch_size:{}".format(args.batchsize))
    print('')
    out = os.path.join('result', args.network)
    batchsize = args.batchsize
    max_epoch = args.epoch

    train_val, _ = mnist.get_mnist(withlabel=False, ndim=3)
    train_val = TransformDataset(train_val, transform)
    #for visualize
    _, test = mnist.get_mnist(withlabel=True, ndim=3)
    test = TransformDataset(test, transform2)
    label1 = 1
    label2 = 5
    test1 = [i[0] for i in test if(i[1]==label1)]
    test2 = [i[0] for i in test if(i[1]==label2)]
    test1 = test1[0:5]
    test2 = test2[5:10]

    if args.network=='conv':
        import Network.mnist_conv as Network
    elif args.network=='fl':
        import Network.mnist_fl as Network
    else:
        raise Exception('Error!')

    Enc = Network.Encoder(n_dimz=args.n_dimz)
    Dec = Network.Decoder(n_dimz=args.n_dimz)
    Critic = Network.Critic()
    chainer.cuda.get_device(args.gpu).use()
    Enc.to_gpu(args.gpu)
    Dec.to_gpu(args.gpu)
    Critic.to_gpu(args.gpu)

    train, valid = split_dataset_random(train_val, 50000, seed=0)

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
    opt_Enc = make_optimizer(Enc)
    opt_Dec = make_optimizer(Dec)
    opt_Critic = make_optimizer(Critic) 
    #trainer
    updater = Updater.ACAIUpdater(
    model=(Enc, Dec, Critic),
    iterator=train_iter,
    optimizer={'Enc':opt_Enc, 'Dec':opt_Dec, 'Critic':opt_Critic},
    n_dimz = args.n_dimz,
    net = args.network,
    device=args.gpu)


    trainer = training.Trainer(updater, (max_epoch, 'epoch'),
     out=out)
    trainer.extend(extensions.LogReport(log_name='log'))
    snapshot_interval = (args.snapshot, 'epoch')
    display_interval = (1, 'epoch')
    trainer.extend(extensions.snapshot_object(Enc,
        filename='Enc_snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(Dec,
        filename='Dec_snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(Critic,
        filename='Critic_snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.PrintReport(['epoch', 'Critic_loss', 'rec_loss']),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(Visualizer.out_generated_image(Enc, Dec, Critic,
        test1, test2, out),
        trigger=(1, 'epoch'))
    trainer.run()
    del trainer

if __name__ == '__main__':
    main()
