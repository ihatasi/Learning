#!/usr/bin/python3

import chainer
import chainer.functions as F
from chainer import Variable


class DCGANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.enc = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)


    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        #G(x, E(x))->1, G(D(z), z)->0
        #real label 1
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        #fake label 0
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        #G(D(z), z)->1
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def loss_enc(self, enc, y_real):
        batchsize = len(y_real)
        #G(x, E(x))->1
        loss = F.sum(F.softplus(y_real)) / batchsize
        chainer.report({'loss': loss}, enc)
        return loss


    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        enc_optimizer = self.get_optimizer('enc')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device))
        xp = chainer.backend.get_array_module(x_real.data)

        gen, dis, enc = self.gen, self.dis, self.enc
        batchsize = len(batch)
        #x_real->enc(x_real)->dis(x_real, enc(x_real))=1
        enc_z = enc(x_real)
        y_rl, y_real = dis(x_real, enc_z)
        #z->gen(z)->dis(gen(x), z)=0
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        y_fl, y_fake = dis(x_fake, z)

        """update optimizers"""
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
        enc_optimizer.update(self.loss_enc, enc, y_real)
