import tensorflow as tf
import numpy as np


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')

        self.v = self.build_net("Critic")
        self.v_target = self.build_net("Target")
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target')
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic')
        self.replace_target_op = [tf.assign(t, p) for t, p in zip(self.t_params, self.params)]

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + 0.8 * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_, r = s[np.newaxis, :], s_[np.newaxis, :], r[np.newaxis, :]

        v_ = self.sess.run(self.v_target, {self.s: s_})
        # v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error

    def build_net(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("dense1"):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=20,  # number of hidden units
                    activation=tf.nn.relu,  # None
                    # have to be linear to make sure the convergence of actor.
                    # But linear approximator seems hardly learns the correct Q.
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='l1'
                )
            with tf.variable_scope("dense2"):
                v = tf.layers.dense(
                    inputs=l1,
                    units=1,  # output units
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='V'
                )

        return v

    def update_target(self):
        self.sess.run(self.replace_target_op)
