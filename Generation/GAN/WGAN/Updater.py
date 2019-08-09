#!/usr/bin/python3

import chainer
import chainer.functions as F
from chainer import Variable


class WGANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.cri = kwargs.pop('models')
        self.iteration = 0
        super(WGANUpdater, self).__init__(*args, **kwargs)


    def loss_cri(self, cri, y_real, y_fake):
        wasserstein_distance = y_real - y_fake
        loss = -wasserstein_distance
        chainer.report({'loss': loss}, cri)
        return loss

    def loss_gen(self, gen, y_fake):
        loss = -y_fake
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        cri_optimizer = self.get_optimizer('cri')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.backend.get_array_module(x_real.data)

        gen, cri = self.gen, self.cri
        batchsize = len(batch)
        #generate
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)

        #critic
        y_real = cri(x_real)
        y_fake = cri(x_fake)

        #update
        cri_optimizer.update(self.loss_dis, cri, y_real, y_fake)
        if self.iteration > 2500 and self.iteration % self.n_c == 0:
            gen_optimizer.update(self.loss_gen, gen, y_fake)
