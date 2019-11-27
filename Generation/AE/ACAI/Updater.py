import chainer
import chainer.functions as F
from chainer import Variable
import random
import numpy as np


class ACAIUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.Enc, self.Dec, self.Critic  = kwargs.pop('model')
        self.gam = 0.2
        self.lamb = 0.5
        self.n_hidden = kwargs.pop('n_dimz')
        self.net = kwargs.pop('net')
        super(ACAIUpdater, self).__init__(*args, **kwargs)

    def loss_Critic(self, dis_c, alpha, dis_y1, dis_y2):
        xp = chainer.backend.get_array_module(dis_c.data)
        batchsize = len(dis_c)
        loss = F.sum((F.squeeze(dis_c)-F.squeeze(alpha))**2) / batchsize
        loss_ = F.sum(dis_y1**2) / batchsize
        loss_ += F.sum(dis_y2**2) / batchsize
        loss += loss_/2
        chainer.report({'Critic_loss':loss})
        return loss

    def loss_Enc(self, x1, x2, y1, y2, dis_c):
        batchsize = len(x1)
        loss = F.mean_squared_error(x1, y1)
        loss += F.mean_squared_error(x2, y2)
        rec_loss = loss
        loss = rec_loss
        closs = (self.lamb*F.sum(dis_c**2)/batchsize)
        loss += closs
        chainer.report({'enc_loss': loss})
        return loss

    def loss_Dec(self, x1, x2, y1, y2, dis_c):
        batchsize = len(x1)
        rec_loss = F.mean_squared_error(x1, y1)
        rec_loss += F.mean_squared_error(x2, y2)
        loss = rec_loss
        closs = (self.lamb*F.sum(dis_c**2)/batchsize)
        loss += closs
        chainer.report({'rec_loss': rec_loss, 'closs':closs})
        return loss

    def update_core(self):
        Enc_optimizer = self.get_optimizer('Enc')
        Dec_optimizer = self.get_optimizer('Dec')
        Critic_optimizer = self.get_optimizer('Critic')

        batch1 = self.get_iterator('main').next()
        batch2 = random.sample(batch1, len(batch1))
        x1 = Variable(self.converter(batch1, self.device))
        x2 = Variable(self.converter(batch2, self.device))
        xp = chainer.backend.get_array_module(x1.data)
        batchsize = len(batch1)
        alpha = chainer.Variable(xp.random.rand(batchsize, dtype=xp.float32))
        alpha = 0.5 - F.absolute(0.5 - alpha)
        if self.net == 'conv':
            alpha = alpha.reshape(batchsize, 1, 1 ,1)
        else:
            alpha = alpha.reshape(batchsize, 1)

        z1 = self.Enc(x1)
        z2 = self.Enc(x2)
        zc =  alpha*z1+(1.0-alpha)*z2
        yc = self.Dec(zc)
        y1 = self.Dec(z1)
        y2 = self.Dec(z2)
        cdis_c = self.Critic(yc)
        cdis_y1 = self.Critic(self.gam*x1+(1-self.gam)*y1)
        cdis_y2 = self.Critic(self.gam*x2+(1-self.gam)*y2)
        Critic_optimizer.update(self.loss_Critic, cdis_c, alpha, cdis_y1, cdis_y2)
        Enc_optimizer.update(self.loss_Enc, x1, x2, y1, y2, cdis_c)
        Dec_optimizer.update(self.loss_Dec, x1, x2, y1, y2, cdis_c)
