#!/usr/bin/python3

import chainer
import chainer.functions as F
from chainer import Variable


class AEUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.model  = kwargs.pop('model')
        super(AEUpdater, self).__init__(*args, **kwargs)

    def loss_AE(self, model, t, y):
        batchsize = len(t)
        loss = F.mean_squared_error(y, t)
        chainer.report({'loss': loss}, model)
        return loss

    def update_core(self):
        optimizer = self.get_optimizer('main')

        batch = self.get_iterator('main').next()
        x = Variable(self.converter(batch, self.device))
        xp = chainer.backend.get_array_module(x.data)
        model = self.model
        y = model(x)

        optimizer.update(self.loss_AE, model, x, y)
