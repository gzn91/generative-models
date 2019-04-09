import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


# orthogonal weight initialization by OpenAI
def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def fc(x, units, name, activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        nin = x.get_shape().as_list()[-1]
        scale = np.sqrt(2. / (nin + units))
        w = tf.get_variable(shape=(nin, units), initializer=tc.layers.xavier_initializer(), name='w')
        b = tf.get_variable(shape=(1, units), initializer=tf.zeros_initializer(), name='b')

        h = tf.matmul(x, w) + b
        h_activ = activation_fn(h)

        return h_activ


def conv2d(x, filters, name, kernel_size=3, strides=1, padding='SAME',
           activation_fn=tf.nn.relu, training=True, use_bn=False):
    nb, nw, nh, nc = x.get_shape().as_list()
    scale = np.sqrt(2. / (nh + nc))
    with tf.variable_scope(name):
        wx = tf.get_variable("wx", [kernel_size, kernel_size, nc, filters], initializer=tc.layers.xavier_initializer_conv2d())
        b = tf.get_variable("b", [1, 1, 1, filters], initializer=tf.constant_initializer(0.))
        x = tf.nn.conv2d(x, filter=wx, strides=[1, strides, strides, 1], padding=padding) + b
        if use_bn:
            x = tf.layers.batch_normalization(x, training=training)
    return activation_fn(x)


def conv2d_transpose(x, filters, name, kernel_size=3, strides=2, padding='SAME',
                     activation_fn=tf.nn.relu, training=True, use_bn=False):
    nb, nw, nh, nc = x.get_shape().as_list()
    scale = np.sqrt(2. / (nh + nc))
    if padding == 'VALID':
        out_h = nh * strides + max(kernel_size - strides, 0)
        out_w = nw * strides + max(kernel_size - strides, 0)
    elif padding == 'SAME':
        out_h = nh * strides
        out_w = nw * strides
    else:
        raise NameError("padding must be 'SAME' or 'VALID'")
    out_shape = [tf.shape(x)[0], out_h, out_w, filters]

    with tf.variable_scope(name):
        wx = tf.get_variable("wx", [kernel_size, kernel_size, filters, nc], initializer=tc.layers.xavier_initializer_conv2d())
        b = tf.get_variable("b", [1, 1, 1, filters], initializer=tf.constant_initializer(0.))
        x = tf.nn.conv2d_transpose(x, filter=wx, output_shape=out_shape,
                                   strides=[1, strides, strides, 1], padding=padding) + b
        if use_bn:
            x = tf.layers.batch_normalization(x, training=training)
    return activation_fn(x)


def max_pool2d(x, name, nkernel, stride, pad='SAME'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, nkernel, nkernel, 1], strides=[1, stride, stride, 1], padding=pad)
