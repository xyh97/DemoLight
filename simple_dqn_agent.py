import tensorflow as tf

from agent import Agent

class SimpleDQNAgent(Agent):
    def construct_weights(self, dim_input, dim_output):
        weights = {}
        weights['w1'] = tf.Variable(tf.glorot_uniform_initializer()([dim_input, self.dic_agent_conf['D_DENSE']]))
        weights['b1'] = tf.Variable(tf.zeros([self.dic_agent_conf['D_DENSE']]))
        for i in range(1, self.dic_agent_conf["N_LAYER"]):
            weights['w' + str(i + 1)] = tf.Variable(tf.glorot_uniform_initializer()([self.dic_agent_conf['D_DENSE'], self.dic_agent_conf['D_DENSE']]))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dic_agent_conf['D_DENSE']]))
        weights['w' + str(self.dic_agent_conf["N_LAYER"] + 1)] = tf.Variable(
            tf.glorot_uniform_initializer()([self.dic_agent_conf['D_DENSE'], dim_output]))
        weights['b' + str(self.dic_agent_conf["N_LAYER"] + 1)] = tf.Variable(
            tf.zeros([dim_output]))

        return weights

    def construct_forward(self, inp, weights, reuse, norm, is_train, prefix='fc'):
        h = self.contruct_layer(tf.matmul(inp, weights['w1']) + weights['b1'],
                                 activation_fn=self._activation_fn, reuse=reuse, is_train=is_train,
                                 norm=norm, scope='1.' + prefix)
        self._h = h
        for i in range(1, self.dic_agent_conf["N_LAYER"]):
            w = weights['w' + str(i + 1)]
            b = weights['b' + str(i + 1)]
            h = self.contruct_layer(tf.matmul(h, w) + b, activation_fn=self._activation_fn,
                                     reuse=reuse, is_train=is_train, norm=norm,
                                     scope=str(i + 1) + '.' + prefix)
        w = weights['w' + str(self.dic_agent_conf["N_LAYER"] + 1)]
        b = weights['b' + str(self.dic_agent_conf["N_LAYER"] + 1)]
        out = tf.matmul(h, w) + b
        return out