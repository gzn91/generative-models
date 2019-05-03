import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from layers import conv2d, conv2d_transpose, fc
from utils import mbgenerator, discriminator_regularizer
import os
from datetime import datetime
file_path = './cifar-10-batches-py'
file_path2 = './train_9'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('train', True, 'If model should be trained.')
flags.DEFINE_bool('restore', False, 'If restore previous model.')
flags.DEFINE_integer('mb_size', 16, 'Size of minibatch')




class GAN(object):

    def __init__(self, img_shape, nd_latent, lr=1e-4, gamma=0.1, flatten=False):

        self.img_ph = tf.placeholder(tf.float32, shape=(None, *img_shape), name='img-ph')

        self._nd_z = nd_latent
        self._img_shape = img_shape
        self._filters = 64
        self._flatten = flatten

        # sample from latent space
        self.z_ph = tf.placeholder(tf.float32, shape=(None, nd_latent), name='z-ph')
        self.is_training = tf.placeholder(tf.bool, ())

        """Generator"""
        recon = self._generator(self.z_ph, 'generator')

        """Discriminator"""
        logits_real, preds_real = self._discriminator(self.img_ph, 'discriminator')

        logits_fake, preds_fake = self._discriminator(recon, 'discriminator', reuse=True)

        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real))
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake))
        d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

        disc_reg = discriminator_regularizer(logits_real, self.img_ph, logits_fake, recon, FLAGS.mb_size)
        d_loss += gamma * disc_reg / 2

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)))

        g_params = tf.trainable_variables(scope='generator')
        d_params = tf.trainable_variables(scope='discriminator')

        # compute gradient
        g_grads = tf.gradients(g_loss, g_params, name='g-grads')
        d_grads = tf.gradients(d_loss, d_params, name='d-grads')
        # g_grads, _grad_norm = tf.clip_by_global_norm(g_grads, 5.0)
        # d_grads, _grad_norm = tf.clip_by_global_norm(d_grads, 5.0)
        g_grads_and_vars = list(zip(g_grads, g_params))
        d_grads_and_vars = list(zip(d_grads, d_params))

        # optimize
        g_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        g_bn_opt = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        d_bn_opt = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        self.g_optimize = [g_optimizer.apply_gradients(g_grads_and_vars), g_bn_opt]
        self.d_optimize = [d_optimizer.apply_gradients(d_grads_and_vars), d_bn_opt]

        self.recon = recon
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.sess = tf.get_default_session()
        self.global_step = tf.Variable(0)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.summary = tf.Summary()
        self.merge_op = tf.summary.merge_all()
        print('Logging to:', './logs/' + str(datetime.now()))
        self.writer = tf.summary.FileWriter('./logs/' + str(datetime.now()))

    def _discriminator(self, img, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            x = img
            h1 = conv2d(x, 128, strides=2, name='down1', training=self.is_training, use_bn=True)
            h2 = conv2d(h1, 256, strides=2, name='down2', training=self.is_training, use_bn=True)
            h3 = conv2d(h2, 512, strides=2, name='down3', training=self.is_training, use_bn=True)

            h4_flat = tf.layers.flatten(h3)

            logits = fc(h4_flat, 1, name='out', activation_fn=lambda x: x)

            return logits, tf.nn.sigmoid(logits)

    def _generator(self, z, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            z = tf.reshape(z, (-1, 1, 1, self._nd_z))

            h1 = conv2d_transpose(z, 1024, name='up1', padding='VALID')
            h2 = conv2d_transpose(h1, 512, name='up2', padding='VALID', training=self.is_training, use_bn=True)
            h3 = conv2d_transpose(h2, 256, name='up3', training=self.is_training, use_bn=True)
            h4 = conv2d_transpose(h3, 128, name='up4', training=self.is_training, use_bn=True)

            recon = conv2d(h4, self._img_shape[-1], kernel_size=1, name='out')

            return tf.nn.sigmoid(recon)

    def train(self, dataset, mb_size=3):
        (x_train, y_train), (x_test, y_test) = dataset()
        x_train, x_test = x_train / 255., x_test / 255.

        data = mbgenerator(x_train, y_train, mb_size, flatten=self._flatten)
        test_data = mbgenerator(x_test, y_test, mb_size, flatten=self._flatten)

        step = self.sess.run(self.global_step)
        while step < 50000:
            imgs, labels = next(data)

            fd_map = {self.img_ph: imgs, self.is_training: True,
                      self.z_ph: np.random.normal(size=(FLAGS.mb_size, self._nd_z))}
            d_loss, _ = self.sess.run([self.d_loss, self.d_optimize], feed_dict=fd_map)
            img, g_loss, _ = self.sess.run([self.recon, self.g_loss, self.g_optimize], feed_dict=fd_map)

            if step % 1000 == 0:
                self.sess.run(self.global_step.assign(step))
                self.save_model()
                plt.imshow(img[0,:,:,0])
                plt.show()
                _str = '*'*20 + f' {step} ' + '*'*20
                print(_str)
                print(f'GLOSS: {g_loss:.6f}\nDLOSS: {d_loss:.6f}')
                print('*'*len(_str))
            step += 1

    def eval(self):
        imgs = self.sess.run(self.recon, feed_dict={self.z_ph: np.random.normal(size=(15, self._nd_z)),
                                                   self.is_training: False})
        for img in imgs:
            plt.imshow(img[:, :, 0])
            plt.show()
        # imageio.imwrite('generated_img.png',imresize(_img, (256,256)))


    def restore_model(self):
        print('RESTORING: {}'.format(tf.train.latest_checkpoint('./saved_model/')))
        self.saver.restore(self.sess, '{}'.format(tf.train.latest_checkpoint('./saved_model/')))

    def save_model(self):
        self.saver.save(self.sess, './saved_model/model', global_step=self.sess.run(self.global_step))


def main(_):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    dataset = tf.keras.datasets.mnist

    np.random.seed(1)

    shape = (28, 28, 1)

    mb_size = FLAGS.mb_size

    with tf.Session(config=config).as_default():

        gan = GAN(img_shape=shape, nd_latent=64)
        if FLAGS.restore:
            gan.restore_model()
        if FLAGS.train:
            gan.train(dataset.load_data, mb_size)
        gan.eval()


tf.app.run()
