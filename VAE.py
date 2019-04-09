import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
from layers import conv2d, conv2d_transpose, fc
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('train', True, 'If model should be trained.')
flags.DEFINE_bool('restore', False, 'If restore previous model.')
flags.DEFINE_integer('mb_size', 32, 'Size of minibatch')


def batchnflat(x, y, mb_size, flatten=False):
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


def log_gaussian(x, mean, stddev=1.0):
    return (-0.5 * np.log(2 * np.pi) - tf.log(stddev) - tf.square(x - mean) /
            (2 * tf.square(stddev)))


class VAE(object):

    def __init__(self, img_shape, nd_latent, lr=4e-4):

        self._nd_obs = np.prod(img_shape)
        self._nd_latent = nd_latent
        self._img_shape = img_shape
        self._filters = 128

        # placeholders
        self.obs_ph = tf.placeholder(tf.float32,shape=(None, self._nd_obs))
        self.target_ph = tf.placeholder(tf.float32, shape=(None, self._nd_obs))
        self.is_training = tf.placeholder(tf.bool, ())

        latentvec = self._encoder(self.obs_ph)

        mean, log_cov = tf.split(value=latentvec, num_or_size_splits=2, axis=-1)

        # Gaussian reparameterization trick
        self.z = mean + tf.exp(.5*log_cov)*tf.random_normal(tf.shape(log_cov))

        logits = self._decoder(self.z)

        kl_loss = 0.5 * tf.reduce_sum(tf.exp(log_cov) + tf.square(mean) - 1.0 - log_cov, axis=1)/self._nd_latent
        # recon_loss = tf.reduce_sum(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_ph, logits=logits), axis=1)/self.nd_obs
        recon_loss = - tf.reduce_sum(log_gaussian(self.target_ph, logits), axis=-1)/self._nd_obs
        elbo = tf.reduce_mean(recon_loss + kl_loss)

        params = tf.trainable_variables()

        # compute gradient
        grads = tf.gradients(elbo, params, name='grads')
        grads, _grad_norm = tf.clip_by_global_norm(grads, 5.0)
        grads_and_vars = list(zip(grads, params))

        # optimize
        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        bn_opt = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optimize = [optimizer.apply_gradients(grads_and_vars, global_step=self.global_step), bn_opt]

        self.recon = tf.reshape(tf.nn.sigmoid(logits),(-1, *img_shape))
        self.latentvec = latentvec
        self.sess = tf.get_default_session()
        self.loss = elbo

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

        self.summary = tf.Summary()
        self.merge_op = tf.summary.merge_all()
        print('Logging to:', './logs/' + str(datetime.now()))
        self.writer = tf.summary.FileWriter('./logs/' + str(datetime.now()))

    # Q(z|x,c)
    def _encoder(self, x):

        with tf.variable_scope('encoder'):

            for i in range(1,5):
                x = conv2d(x, self._filters, strides=2, name='conv{i}', training=self.is_training, use_bn=True)
                self._filters *= 2
            x_flat = tf.layers.flatten(x)

            latentvec = fc(x_flat, units=self._nd_latent*2, activation_fn=lambda x: x, name='latentvec')

            return latentvec

    # P(x|z,c)
    def _decoder(self, z):

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):

            z = fc(z, 3 * 3 * self._nd_latent, 'fc-z')
            z = tf.reshape(z, (-1, 3, 3, self._nd_latent))

            h1 = conv2d_transpose(z, 256, name='up1', padding='VALID', training=self.is_training, use_bn=True)

            h2 = conv2d_transpose(h1, 128, name='up2', training=self.is_training, use_bn=True)

            h3 = conv2d_transpose(h2, 64, name='up3')

            recon = conv2d(h3, self._img_shape[-1], kernel_size=1, name='out')

            return tf.nn.sigmoid(recon)

    def train(self, dataset, mb_size=3):

        (x_train, y_train), (x_test, y_test) = dataset()
        x_train, x_test = x_train/255., x_test/255.

        data = batchnflat(x_train, y_train, mb_size)
        test_data = batchnflat(x_test, y_test, mb_size)

        step = self.sess.run(self.global_step)
        while step < 50000:
            obs, labels = next(data)
            fd_map = {self.obs_ph: obs, self.target_ph: obs, self.is_training: True}
            latentvec, img, loss, _ = self.sess.run([self.latentvec, self.recon, self.loss, self.optimize], feed_dict=fd_map)
            if step % 1000 == 0:
                self.save_model()
                plt.imshow(img[0,:,:,0])
                plt.title(f'target is {labels[0]}')
                plt.show()
                _str = '*'*20 + f' {step} ' + '*'*20
                print(_str)
                print(f'LOSS: {loss:.2f}')
                print('*'*len(_str))
            step += 1

    def eval(self):
        z = np.random.normal(size=(10, self.nd_latent))
        img = self.sess.run(self.recon, feed_dict={self.z: z, self.is_training: False})
        for _img in img:
            plt.imshow(_img[...,0])
            plt.show()

        # imageio.imwrite('generated_img.png',imresize(_img, (256,256)))

    def restore_model(self):
        print('RESTORING: {}'.format(tf.train.latest_checkpoint('./saved_model/')))
        self.saver.restore(self.sess, '{}'.format(tf.train.latest_checkpoint('./saved_model/')))

    def save_model(self):
        self.saver.save(self.sess, './saved_model/model', global_step=self.sess.run(self.global_step))


def main(_):

    dataset = tf.keras.datasets.mnist

    np.random.seed(1)

    shape = (28, 28, 1)

    mb_size = FLAGS.mb_size

    with tf.Session().as_default():

        vae = VAE(shape, nd_latent=128)
        if FLAGS.restore:
            vae.restore_model()
        if FLAGS.train:
            vae.train(dataset.load_data, mb_size)
        vae.eval()

tf.app.run()
