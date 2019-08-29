#!/usr/bin/python3
#https://github.com/pfnet-research/chainer-gan-lib/blob/master/wgan_gp/updater.py

import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable
from chainer.dataset import convert

class WGANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.lam = kwargs.pop('lam')
        self.iteration = 0
        super(WGANUpdater, self).__init__(*args, **kwargs)

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(-y_fake)/batchsize
        chainer.reporter.report({'loss': loss}, gen)
        return loss
    def loss_dis(self, dis, y_real, y_fake, x_real, x_fake):
        batchsize = len(y_fake)
        xp = dis.xp

        eps = xp.random.uniform(0, 1, size=batchsize)\
            .astype("f")[:, None, None, None]
        x_mid = eps * x_real + (1.0 - eps) * x_fake

        y_mid,_ = self.dis(x_mid)
        grad, = chainer.grad([y_mid], [x_mid], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        loss_grad = self.lam * F.mean_squared_error(grad, 
            xp.ones_like(grad.data))

        loss = F.sum(-y_real) / batchsize
        loss += F.sum(y_fake) / batchsize
        wasserstein_distance = -loss
        loss += loss_grad
        chainer.reporter.report({'wasserstein_distance': wasserstein_distance,
        'loss_grad':loss_grad})
        chainer.reporter.report({'loss': loss}, dis)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            x = []
            for j in range(batchsize):
                x.append(np.asarray(batch[j]).astype("f"))
            x_real = Variable(xp.asarray(x))
            y_real,_ = self.dis(x_real)

            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake = self.gen(z)
            y_fake,_ = self.dis(x_fake)

            if i == 0:
                gen_optimizer.update(self.loss_gen, self.gen, y_fake)
            x_fake.unchain_backward()

            dis_optimizer.update(self.loss_dis, self.dis, 
                y_real, y_fake, x_real, x_fake)
