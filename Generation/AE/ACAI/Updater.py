import chainer
import chainer.functions as F
from chainer import Variable
import random
import numpy as np


class ACAIUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.AE, self.Critic  = kwargs.pop('model')
        self.gam = 0.2
        self.lamb = 0.5
        super(ACAIUpdater, self).__init__(*args, **kwargs)

    def loss_Critic(self, dis_c, alpha, dis_y1, dis_y2):
        xp = chainer.backend.get_array_module(dis_c.data)
        batchsize = len(dis_c)
        #alpha = alpha[:, 0:1]
        loss = F.sum((F.squeeze(dis_c)-F.squeeze(alpha))**2) / batchsize
        loss_ = F.sum(dis_y1**2) / batchsize
        loss_ += F.sum(dis_y2**2) / batchsize
        loss += loss_/2
        chainer.report({'Critic_loss':loss})
        return loss

    def loss_AE(self, x1, x2, y1, y2, dis_c):
        batchsize = len(x1)
        loss = F.mean_squared_error(x1, y1)
        loss += F.mean_squared_error(x2, y2)
        rec_loss = loss/2
        loss = rec_loss
        loss += self.lamb*F.sum(dis_c**2)/batchsize
        chainer.report({'AE_loss': loss, 'rec_loss':rec_loss})
        return loss

    def update_core(self):
        AE_optimizer = self.get_optimizer('AE')
        Critic_optimizer = self.get_optimizer('Critic')

        batch = self.get_iterator('main').next()
        batch2 = random.sample(batch, len(batch))
        x1 = Variable(self.converter(batch, self.device))
        x2 = Variable(self.converter(batch2, self.device))

        xp = chainer.backend.get_array_module(x1.data)
        y1, y2, yc, alpha, _, _ = self.AE(x1, x2)
        dis_c = self.Critic(yc)
        dis_y1 = self.Critic(self.gam*x1+(1-self.gam)*y1)
        dis_y2 = self.Critic(self.gam*x2+(1-self.gam)*y2)
        Critic_optimizer.update(self.loss_Critic, dis_c, alpha, dis_y1, dis_y2)
        AE_optimizer.update(self.loss_AE, x1, x2, y1, y2, dis_c)
