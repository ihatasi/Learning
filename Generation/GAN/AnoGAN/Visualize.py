#https://github.com/chainer/chainer/blob/v5/examples/dcgan/visualize.py
#!/usr/bin/python3

import os
import chainer
import numpy as np
import chainer.backends.cuda
import matplotlib.pyplot as plt
from PIL import Image
from chainer import Variable


def out_generated_image(gen, dis, ser, valid, out, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        img = valid[0].reshape(28, 28)
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(1)))
        with chainer.using_config('train', False):
            x = gen(ser(z))
        x = chainer.backends.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        x = x.reshape(28, 28)
        imgs = np.concatenate((img, x), axis=1)

        preview_dir = '{}/{}/preview'.format(out, dst)
        preview_path = preview_dir +\
            '/image{:0>4}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.imshow(imgs, cmap='gray')
        plt.savefig(preview_path)
    return make_image
