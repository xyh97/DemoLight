import tensorflow as tf
import numpy as np


class Critic(object):
    def __init__(self, sess, n_features, alpha=0.1, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.q_ = tf.placeholder(tf.float32, [None, 8], "q_")
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a = tf.placeholder(tf.int32, [None, 1], 'act')
        self.act_probs = tf.placeholder(tf.float32, [None, 8], 'probs')

        self.q = self.build_net("Critic")
        self.q_target = self.build_net("Target")
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target')
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic')
        self.replace_target_op = [tf.assign(t, p) for t, p in zip(self.t_params, self.params)]

        self.q_a = tf.batch_gather(self.q, self.a)

        # self.v = alpha * tf.log(tf.reduce_sum(tf.exp(self.q/alpha), axis=1, keepdims=True))
        # self.v = tf.reduce_sum(self.act_probs * self.q, axis=1, keepdims=True)
        self.v_target = tf.reduce_max(self.q_target, axis=1, keepdims=True)

        with tf.variable_scope('squared_TD_error'):
            # self.td_error = self.r + 0.8 * self.v_ - self.v
            # self.td_error = self.q_a - (self.r + 0.8 * self.v_)
            self.td_error = self.q_a - self.v_
            # self.h = -tf.reduce_sum(self.act_probs * tf.log(self.act_probs), axis=1, keepdims=True)
            # self.error = self.v - (self.r + 0.8 * self.v_ + alpha * self.h)
            self.error = self.q_a - (self.r + 0.8 * tf.reduce_max(self.q_, axis=1, keepdims=True))
            self.loss = tf.reduce_mean(0.5*tf.square(self.error))  # TD_error = (r+gamma*V_next) - V_eval
            # self.loss = tf.reduce_mean(0.5*tf.square(self.error))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_, a, act_probs):
        s, s_, r, a = s[np.newaxis, :], s_[np.newaxis, :], r[np.newaxis, :], a[np.newaxis, :]

        v_, q_ = self.sess.run([self.v_target, self.q_target], {self.s: s_, self.act_probs: act_probs})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r, self.a: a, self.q_: q_})
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
                q = tf.layers.dense(
                    inputs=l1,
                    units=8,  # output units
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='V'
                )

        return q

    def update_target(self):
        self.sess.run(self.replace_target_op)
