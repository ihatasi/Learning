#!/usr/bin/python3

import chainer
import chainer.functions as F
from chainer import Variable


class zizUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.enc = kwargs.pop('models')
        self.n_dimz = kwargs.pop("n_dimz")
        self.iteration = 0
        super(zizUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, z, re_z):
        loss = F.mean_squared_error(z, re_z)
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
        #gen.disable_update()
        #dis.disable_update()

        #z->gen(z)->enc(z)
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        re_z = enc(x_fake)
        enc_optimizer.update(self.loss_enc, enc, z, re_z)

class iziUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.enc = kwargs.pop('models')
        self.n_dimz = kwargs.pop("n_dimz")
        self.iteration = 0
        super(iziUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_real, re_x):
        loss = F.mean_squared_error(x_real, re_x)
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
        #gen.disable_update()
        #dis.disable_update()

        #x gen(enc(x))
        z = enc(x_real)
        re_x = gen(z)
        enc_optimizer.update(self.loss_enc, enc, x_real, re_x)

class izifUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis, self.enc = kwargs.pop('models')
        self.n_dimz = kwargs.pop("n_dimz")
        self.iteration = 0
        super(izifUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_real, re_x, f_x, f_re_x):
        k=1
        loss = F.mean_squared_error(x_real, re_x)
        loss += F.mean_squared_error(f_x, f_re_x)
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
        #gen.disable_update()
        #dis.disable_update()

        #x_real gen(enc(x_real)) + f(x_real) f(gen(enc(x_real)))
        z = enc(x_real)
        re_x = gen(z)
        _,f_x = dis(x_real)
        _,f_re_x = dis(re_x)
        enc_optimizer.update(self.loss_enc, enc, x_real, re_x, f_x, f_re_x)
