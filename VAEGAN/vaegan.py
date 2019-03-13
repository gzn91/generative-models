import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from utils import mbgenerator, discriminator_regularizer
from matplotlib import pyplot as plt
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('train', True, 'If model should be trained.')
flags.DEFINE_bool('restore', False, 'If restore previous model.')
flags.DEFINE_integer('mb_size', 32, 'Size of minibatch')


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


# log-likelihood of d-dim iid unit var gaussian
def log_gaussian(x, mean, stddev=1.0):
    return (-0.5 * np.log(2 * np.pi) - tf.log(stddev) - tf.square(x - mean) /
            (2 * tf.square(stddev)))


class CVAEGAN(object):

    def __init__(self, img_shape, nd_latent, lr=1e-4, lam=.5, gamma=.01, flatten=True):

        self.nd_obs = np.prod(img_shape)
        self.nd_latent = nd_latent
        self.nd_l = 16
        self.img_shape = img_shape
        self.flatten = flatten

        # placeholders
        self.obs_ph = tf.placeholder(tf.float32,shape=(None, self.nd_obs))
        self.z_ph = tf.placeholder(tf.float32, shape=(None, self.nd_latent))
        self.condition_ph = tf.placeholder(tf.int32, shape=(None,))
        condition_ph = tf.to_float(tf.one_hot(self.condition_ph, 10))  # categorical dist
        self.target_ph = tf.placeholder(tf.float32, shape=(None, self.nd_obs))

        latentvec = self._encoder(self.obs_ph, condition_ph)

        mean, log_cov = tf.split(value=latentvec, num_or_size_splits=2, axis=-1)

        # Gaussian reparameterization trick
        self.z = mean + tf.exp(.5*log_cov)*tf.random_normal(tf.shape(log_cov))

        # Auxiliary z (Zp)
        aux_z = tf.random_normal(shape=tf.shape(self.z))

        obs_recon, _ = self._decoder(self.z, condition_ph)
        aux_obs_recon, _ = self._decoder(aux_z, condition_ph)

        disl_fake, logits_fake = self.discriminator(obs_recon, condition_ph)
        disl_real, logits_real = self.discriminator(self.obs_ph, condition_ph)
        _, logits_aux = self.discriminator(aux_obs_recon, condition_ph)

        # cvae prior loss
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(log_cov) + tf.square(mean) - 1.0 - log_cov, axis=1)/self.nd_latent

        # cvae recon loss
        recon_loss = - tf.reduce_sum(log_gaussian(tf.stop_gradient(disl_real), disl_fake), axis=1)/self.nd_l

        # discriminator loss
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real))
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake))
        d_loss_aux = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_aux, labels=tf.zeros_like(logits_aux))

        # generator loss
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))
        g_loss_aux = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_aux, labels=tf.ones_like(logits_fake))

        discriminator_loss = tf.reduce_mean(d_loss_aux + d_loss_fake + d_loss_real)/3

        disc_reg = discriminator_regularizer(logits_real, self.obs_ph, logits_fake, obs_recon, FLAGS.mb_size)
        discriminator_loss += gamma * disc_reg / 2

        encoder_loss = tf.reduce_mean(recon_loss+kl_loss)

        generator_loss = tf.reduce_mean(lam*recon_loss+.5*(g_loss_aux+g_loss))

        # params
        e_params = tf.trainable_variables(scope='encoder')
        g_params = tf.trainable_variables(scope='decoder')
        d_params = tf.trainable_variables(scope='discriminator')


        # optimizers
        self.global_step = tf.train.get_or_create_global_step()
        e_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # compute gradient
        e_grads = tf.gradients(encoder_loss, e_params, name='e-grads')
        # e_grads, _grad_norm = tf.clip_by_global_norm(e_grads, 1.0)
        g_grads = tf.gradients(generator_loss, g_params, name='g-grads')
        # g_grads, _grad_norm = tf.clip_by_global_norm(g_grads, 1.0)
        d_grads = tf.gradients(discriminator_loss, d_params, name='d-grads')
        # d_grads, _grad_norm = tf.clip_by_global_norm(d_grads, 1.0)
        e_grads_and_vars = list(zip(e_grads, e_params))
        g_grads_and_vars = list(zip(g_grads, g_params))
        d_grads_and_vars = list(zip(d_grads, d_params))

        self.optimize_enc = e_optimizer.apply_gradients(e_grads_and_vars, global_step=self.global_step)
        self.optimize_gen = g_optimizer.apply_gradients(g_grads_and_vars)
        self.optimize_dis = d_optimizer.apply_gradients(d_grads_and_vars)

        self.recon = tf.reshape(obs_recon, (-1, *img_shape))
        self.latentvec = latentvec
        self.sess = tf.get_default_session()
        self.encoder_loss = encoder_loss
        self.generator_loss = generator_loss
        self.dis_loss = discriminator_loss

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

        self.summary = tf.Summary()
        self.merge_op = tf.summary.merge_all()
        print('Logging to:', './logs/' + str(datetime.now()))
        self.writer = tf.summary.FileWriter('./logs/' + str(datetime.now()))

    def _fc(self, x, nh, scope, activ=tf.nn.elu):
        with tf.variable_scope(scope):
            nin = x.get_shape().as_list()[-1]
            scale = np.sqrt(2. / (nin + nh))
            w = tf.get_variable(shape=(nin, nh), initializer=tf.random_normal_initializer(stddev=scale), name='w')
            b = tf.get_variable(shape=(1, nh), initializer=tf.zeros_initializer(), name='b')

            h = tf.matmul(x, w) + b
            h_activ = activ(h)

            return h_activ

    # Q(z|x,c)
    def _encoder(self, x, c):

        with tf.variable_scope('encoder'):
            xc = tf.concat([x, c], axis=-1)
            with tf.variable_scope('block1'):
                x = self._fc(xc, 512, scope='fc1')
                x = self._fc(x, 256, scope='fc2')
                x = self._fc(x, 128, scope='fc3')

            latentvec = self._fc(x, nh=self.nd_latent*2, activ=lambda x: x, scope='latentvec')

            return latentvec

    # P(x|z,c)
    def _decoder(self, z, c):

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):

            c = self._fc(c, nh=64, scope='init-cfc')
            z = self._fc(z, nh=64, scope='init-zfc')
            x = tf.concat([z, c], axis=-1)
            x = self._fc(x, 256, scope='fc1')
            x = self._fc(x, 512, scope='fc2')

            x = self._fc(x, nh=self.nd_obs, activ=lambda x: x, scope='p_xz')

            return tf.nn.sigmoid(x), x

    # D(x)
    def discriminator(self, x, c):

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            xc = tf.concat([x, c], axis=-1)
            x = self._fc(xc, 512, scope='fc1')
            x = self._fc(x, 128, scope='fc2')

            with tf.variable_scope('layer_l'):
                dis_l = self._fc(x, self.nd_l, activ=lambda x: x, scope='fcl')

            logits = self._fc(tf.nn.elu(dis_l), nh=1, activ=lambda x: x, scope='logits')

            return dis_l, logits

    def train(self, dataset, mb_size=3):

        (x_train, y_train), (x_test, y_test) = dataset()
        x_train, x_test = x_train/255., x_test/255.

        data = mbgenerator(x_train, y_train, mb_size, flatten=self.flatten)
        test_data = mbgenerator(x_test, y_test, mb_size, flatten=self.flatten)

        step = self.sess.run(self.global_step)
        while step < 50000:
            obs, labels = next(data)
            fd_map = {self.obs_ph: obs, self.condition_ph: labels}
            # opt encoder
            enc_loss, _ = self.sess.run([self.encoder_loss, self.optimize_enc], feed_dict=fd_map)
            # opt decoder
            img, gen_loss, _ = self.sess.run([self.recon, self.generator_loss, self.optimize_gen], feed_dict=fd_map)
            # opt discriminator
            dis_loss, _ = self.sess.run([self.dis_loss, self.optimize_dis], feed_dict=fd_map)

            if step % 1000 == 0:
                self.save_model()
                plt.imshow(img[0,:,:,0])
                plt.title(f'target is {labels[0]}')
                plt.show()
                _str = '*'*20 + f' {step} ' + '*'*20
                print(_str)
                print(f'DIS_LOSS: {dis_loss:.2f}')
                print(f'GEN_LOSS: {gen_loss:.2f}')
                print(f'ENC_LOSS: {enc_loss:.2f}')
                print('*'*len(_str))
            step += 1

    def eval(self):
        z = np.random.normal(size=(10, self.nd_latent))
        c = np.arange(0,10).reshape((10,))
        img = self.sess.run(self.recon, feed_dict={self.z: z, self.condition_ph: c})
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

        cvae = CVAEGAN(shape, nd_latent=32)
        if FLAGS.restore:
            cvae.restore_model()
        if FLAGS.train:
            cvae.train(dataset.load_data, mb_size)
        cvae.eval()

tf.app.run()