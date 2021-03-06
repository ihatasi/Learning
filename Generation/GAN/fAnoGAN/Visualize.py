#https://github.com/chainer/chainer/blob/v5/examples/dcgan/visualize.py
#!/usr/bin/python3

import os
import chainer
import numpy as np
import chainer.backends.cuda
from PIL import Image
from chainer import Variable


def out_generated_image(gen, enc, rows, cols, seed, out, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))


        with chainer.using_config('train', False):
            x = gen(z)
            y = gen(enc(x))
        x = chainer.backends.cuda.to_cpu(y.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        if dst == "mnist":
            x = x.reshape((rows, cols, 1, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W))
        else:
            x = x.reshape((rows, cols, 3, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, 3))


        preview_dir = '{}/{}/preview'.format(out, dst)
        preview_path = preview_dir +\
            '/epoch_{:0>4}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image
