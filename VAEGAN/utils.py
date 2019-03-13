import numpy as np
import tensorflow as tf


def mbgenerator(x, y, mb_size, flatten=False):
    inds = np.arange(x.shape[0])
    np.random.shuffle(inds)
    while True:
        ind = np.random.choice(inds, mb_size, replace=False)
        x_mb = x[ind]
        y_mb = y[ind].reshape(mb_size, )
        if flatten:
            x_mb = np.reshape(x_mb, (mb_size, np.prod(x[0].shape)))
        else:
            x_mb = x_mb[..., None]

        yield x_mb, y_mb


def kl_from_hist(p, q):
    q[q == 0] = 1e-8
    q = q/np.sum(q)
    return np.sum(np.where(p != 0., p * np.log(p / q), 0.0))


# https://github.com/rothk/Stabilizing_GANs # added batch_size arg
def discriminator_regularizer(D1_logits, D1_arg, D2_logits, D2_arg, batch_size):
    D1 = tf.nn.sigmoid(D1_logits)
    D2 = tf.nn.sigmoid(D2_logits)
    grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
    grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [batch_size,-1]), axis=1, keepdims=False)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [batch_size,-1]), axis=1, keepdims=False)

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer
