import chainer
import chainer.functions as F
import chainer.links as L

class AE(chainer.Chain):

    def __init__(self, n_dimz, n_out=784):
        super(AE.self).__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, self.n_dimz)
            self.l2 = L.Linear(None, self.n_dimz)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h
