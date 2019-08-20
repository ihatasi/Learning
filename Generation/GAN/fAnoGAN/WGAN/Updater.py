#!/usr/bin/python3
import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable
from chainer import optimizers, Variable, training
from chainer import iterators, datasets, serializers
from chainer.dataset import iterator as iterator_module
from chainer.dataset import convert



class WGANUpdater(training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.iteration = 0
        super(WGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_real, y_fake):
        batchsize = len(y_fake)
        loss = -F.sum(y_real - y_fake) / batchsize
        wasserstein_distance = -loss
        chainer.reporter.report({'loss': loss}, dis)
        chainer.reporter.report({'wasserstein distance': wasserstein_distance})
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(-y_fake) / batchsize
        chainer.reporter.report({'loss': loss}, gen)
        return loss


    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        # read data
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x_real = Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x_real)

        #Generate
        z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        x_fake = self.gen(z)

        #Critic
        y_real = self.dis(x_real)
        y_fake = self.dis(x_fake)

        #Update critic
        dis_optimizer.update(self.loss_dis, self.dis, y_real, y_fake)
        #Update generator
        if self.iteration < 2500 and self.iteration % 100 == 0:
            gen_optimizer.update(self.loss_gen, self.gen, y_fake)
        if self.iteration > 2500 and self.iteration % self.n_dis == 0:
            gen_optimizer.update(self.loss_gen, self.gen, y_fake)
