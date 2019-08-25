import tensorflow as tf
import numpy as np


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.q_a_ = tf.placeholder(tf.float32, [None, 1], "q_next")
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a = tf.placeholder(tf.int32, [None, 1], 'act')
        self.act_probs = tf.placeholder(tf.float32, [None, 8], 'act_probs')

        self.v = self.build_net("Value")
        self.q = self.build_q_net("Q-Value")
        self.q_a = tf.batch_gather(self.q, self.a)
        # self.v = tf.reduce_sum(self.q * self.act_probs, axis=1, keepdims=True)
        # self.v_target = self.build_net("Target")
        # self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target')
        # self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic')
        # self.replace_target_op = [tf.assign(t, p) for t, p in zip(self.t_params, self.params)]

        with tf.variable_scope('squared_TD_error'):
            # self.td_error = self.r + 0.8 * self.v_ - self.v
            self.td_error = self.q_a - self.v
            q_loss = tf.reduce_mean(tf.squared_difference(self.q_a, self.r + 0.8 * self.q_a_))
            v_loss = tf.reduce_mean(tf.squared_difference(self.v, self.r + 0.8 * self.v_))
            # self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
            self.loss = q_loss + v_loss
            # self.loss = q_loss
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_, a, next_a):
        s, s_, r = s[np.newaxis, :], s_[np.newaxis, :], r[np.newaxis, :]
        a, next_a = a[np.newaxis, :], next_a[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        q_a_ = self.sess.run(self.q_a, {self.s: s_, self.a: next_a})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s,  self.r: r, self.v_: v_,
                                     self.q_a_: q_a_, self.a: a})
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


    def build_q_net(self, scope):
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
                    name='q'
                )

        return q

    # def update_target(self):
    #     self.sess.run(self.replace_target_op)