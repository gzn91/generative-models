import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from layers import conv2d, conv2d_transpose, fc
from utils import mbgenerator, mse, pull_away_term
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('train', True, 'If model should be trained.')
flags.DEFINE_bool('restore', False, 'If restore previous model.')
flags.DEFINE_integer('mb_size', 3, 'Size of minibatch')


class EBGAN(object):

    def __init__(self, img_shape, nd_latent, lr=4e-4, margin=5.0, pt_loss_weight=0.1, flatten=False):

        self.img_ph = tf.placeholder(tf.float32, shape=(None, *img_shape))

        self._zdim = nd_latent
        self._img_shape = img_shape
        self._flatten = flatten
        self.global_step = tf.train.get_or_create_global_step()

        margin = tf.train.exponential_decay(margin, self.global_step, 10000, .7)

        # sample from latent space
        self.z_ph = tf.placeholder(tf.float32, (None, self._zdim))
        self.is_training = tf.placeholder(tf.bool, ())

        """Generator"""
        grecon = self._generator(self.z_ph, 'generator')

        """Discriminator"""
        _, logits_real, drecon_real = self._discriminator(self.img_ph, 'discriminator')

        z_fake, logits_fake, drecon_fake = self._discriminator(grecon, 'discriminator', reuse=True)

        # Discriminator loss
        d_loss_real = mse(self.img_ph - drecon_real, FLAGS.mb_size)
        d_loss_fake = mse(grecon - drecon_fake, FLAGS.mb_size)
        hinge_loss = tf.nn.relu(margin - d_loss_fake)
        d_loss = d_loss_real + hinge_loss

        # Generator loss
        pt_loss = pull_away_term(z_fake, FLAGS.mb_size)
        g_loss = d_loss_fake + pt_loss_weight * pt_loss

        g_params = tf.trainable_variables(scope='generator')
        d_params = tf.trainable_variables(scope='discriminator')

        # compute gradient
        g_grads = tf.gradients(g_loss, g_params, name='g-grads')
        d_grads = tf.gradients(d_loss, d_params, name='d-grads')
        # d_grads = tf.cond(d_loss > .2, lambda: d_grads, lambda: [tf.zeros_like(_) for _ in d_grads])

        g_grads_and_vars = list(zip(g_grads, g_params))
        d_grads_and_vars = list(zip(d_grads, d_params))

        # optimize
        g_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=.5*lr)
        g_bn_opt = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        d_bn_opt = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        self.g_optimize = [g_optimizer.apply_gradients(g_grads_and_vars, global_step=self.global_step), g_bn_opt]
        self.d_optimize = [d_optimizer.apply_gradients(d_grads_and_vars), d_bn_opt]

        self.grecon = grecon
        self.drecon = drecon_real
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.sess = tf.get_default_session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.summary = tf.Summary()
        self.merge_op = tf.summary.merge_all()
        print('Logging to:', './logs/' + str(datetime.now()))
        self.writer = tf.summary.FileWriter('./logs/' + str(datetime.now()))

    def _encoder(self, img, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            h1 = conv2d(img, 16, strides=2, name='down1', training=self.is_training, use_bn=True)

            h2 = conv2d(h1, 64, name='down2', training=self.is_training, use_bn=True)

            h3 = conv2d(h2, 128, strides=2, name='down3', training=self.is_training, use_bn=True)
            h3_flat = tf.layers.flatten(h3)

            z = fc(h3_flat, self._zdim, 'out', activation_fn=tf.tanh)

            return z

    def _generator(self, z, scope, reuse=False, activation_fn=tf.nn.sigmoid):

        with tf.variable_scope(scope, reuse=reuse):
            z = fc(z, 3 * 3 * self._zdim, 'fc-z')
            z = tf.reshape(z, (-1, 3, 3, self._zdim))

            h1 = conv2d_transpose(z, 256, name='up1', padding='VALID', training=self.is_training, use_bn=True)

            h2 = conv2d_transpose(h1, 128, name='up2', training=self.is_training, use_bn=True)

            h3 = conv2d_transpose(h2, 64, name='up3', training=self.is_training, use_bn=True)

            recon = conv2d(h3, self._img_shape[-1], 'out', kernel_size=1, activation_fn=lambda x: x)
            return activation_fn(recon)

    def _discriminator(self, img, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            z = self._encoder(img, 'encoder')
            recon = self._generator(z, 'decoder', activation_fn=lambda x: x)

        return z, recon, tf.sigmoid(recon)

    def train(self, dataset, mb_size=3):
        (x_train, y_train), (x_test, y_test) = dataset()
        x_train, x_test = x_train / 255., x_test / 255.

        data = mbgenerator(x_train, y_train, mb_size, flatten=self._flatten)
        test_data = mbgenerator(x_test, y_test, mb_size, flatten=self._flatten)

        step = self.sess.run(self.global_step)
        while step < 50000:
            imgs, labels = next(data)

            fd_map = {self.img_ph: imgs, self.is_training: True,
                      self.z_ph: np.random.normal(size=(imgs.shape[0],self._zdim))}
            dimg, d_loss, _ = self.sess.run([self.drecon, self.d_loss, self.d_optimize], feed_dict=fd_map)
            gimg, g_loss, _ = self.sess.run([self.grecon, self.g_loss, self.g_optimize], feed_dict=fd_map)

            if step % 1000 == 0:
                self.save_model()
                plt.subplot(1, 2, 1)
                plt.imshow(gimg[0, :, :, 0])
                plt.title('generator')
                plt.subplot(1, 2, 2)
                plt.imshow(dimg[0, :, :, 0])
                plt.title('discriminator')
                plt.show()
                _str = '*' * 20 + f' {step} ' + '*' * 20
                print(_str)
                print(f'GLOSS: {g_loss:.6f}\nDLOSS: {d_loss:.6f}')  # \nMEAN: {mean}\n STDDEV: {np.exp(log_stddev)}')
                print('*' * len(_str))
            step += 1

    def eval(self):
        imgs = self.sess.run(self.grecon, feed_dict={self.z_ph: np.random.normal(size=(15, self._zdim)),
                                                     self.is_training: False})

        for img in imgs:
            plt.imshow(img[..., 0])
            plt.show()

    def restore_model(self):
        print('RESTORING: {}'.format(tf.train.latest_checkpoint('./saved_model/')))
        self.saver.restore(self.sess, '{}'.format(tf.train.latest_checkpoint('./saved_model/')))

    def save_model(self):
        self.saver.save(self.sess, './saved_model/model', global_step=self.sess.run(self.global_step))


def main(_):
    dataset = tf.keras.datasets.mnist

    np.random.seed(2)

    shape = (28, 28, 1)

    mb_size = FLAGS.mb_size

    with tf.Session().as_default():

        gan = EBGAN(img_shape=shape, nd_latent=64)
        if FLAGS.restore:
            gan.restore_model()
        if FLAGS.train:
            gan.train(dataset.load_data, mb_size)
        gan.eval()


tf.app.run()
