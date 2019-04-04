#!/usr/bin/python3

import chainer
import copy
import chainer.functions as F
from chainer import Variable, reporter, function
from chainer.training  import extensions


class AEEvaluator(extensions.Evaluator):

    def __init__(self, *args, **kwargs):
        #self._target = kwargs.pop('target')
        super(AEEvaluator, self).__init__(*args, **kwargs)

    def loss_AE(self, model, t, y):
        #batchsize = len(t)
        loss = F.mean_squared_error(y, t)
        chainer.report({'loss': loss}, model)
        return loss

    def evaluate(self):
        iterator = self._iterators['main']
        self.target = self.get_target('main')

        it = copy.copy(iterator)
        summary = reporter.DictSummary()
        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                xp = chainer.backend.get_array_module(in_arrays)
                with function.no_backprop_mode():
                    y = self.target(in_arrays)
                    self.loss = self.loss_AE(self.target, in_arrays, y)
            summary.add(observation)
        return summary.compute_mean()
