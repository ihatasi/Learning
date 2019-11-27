import chainer
import chainer.functions as F
from chainer import Variable
import random
import numpy as np


class AEUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.AE = kwargs.pop('model')
        super(AEUpdater, self).__init__(*args, **kwargs)


    def loss_AE(self, x1, x2, y1, y2):
        batchsize = len(x1)
        loss = F.mean_squared_error(x1, y1)
        loss += F.mean_squared_error(x2, y2)
        loss = loss
        chainer.report({'AE_loss': loss})
        return loss

    def update_core(self):
        AE_optimizer = self.get_optimizer('AE')

        batch = self.get_iterator('main').next()
        batch2 = random.sample(batch, len(batch))
        x1 = Variable(self.converter(batch, self.device))
        x2 = Variable(self.converter(batch2, self.device))

        xp = chainer.backend.get_array_module(x1.data)
        y1, y2, _, _ = self.AE(x1, x2)
        AE_optimizer.update(self.loss_AE, x1, x2, y1, y2)
