import tensorflow as tf
import numpy as np


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=True):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def relation(lane_phase_info):
    relations = []
    num_phase = len(lane_phase_info["phase"])
    map = lane_phase_info["phase_roadLink_mapping"]
    for p1 in lane_phase_info["phase"]:
        zeros = [0] * (num_phase - 1)
        count = 0
        for p2 in lane_phase_info["phase"]:
            if p1 == p2:
                continue
            if len(set(map[p1] + map[p2])) != len(map[p1]) + len(map[p2]):
                zeros[count] = 1
            count += 1
        relations.append(zeros)
    relations = np.array(relations).reshape((1, num_phase, num_phase - 1))

    constant = relations
    return constant


class Actor(object):
    def __init__(self, sess, n_features, n_actions, dic_traffic_env_conf, lr=0.001):
        self.sess = sess
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self._activation_fn = tf.nn.leaky_relu

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.int32, [None, 1], "act")
        self.td_error = tf.placeholder(tf.float32, [None, 1], "td_error")  # TD_error
        self.expert_action = tf.placeholder(tf.float32, [None, 8], "expert_action")
        self.batch_size = 32
        #
        # self.acts_prob = self.construct_forward(self.s, True, 'None', True, "Actor", prefix='fc')

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            # log_prob = tf.log(self.acts_prob[0, self.a])
            log_prob = tf.log(tf.batch_gather(self.acts_prob, self.a))
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        self.action = gumbel_softmax(logits=self.acts_prob, temperature=1, hard=False)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

        self.pretrain_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.action, labels=self.expert_action)
        self.pretrain_op = tf.train.AdamOptimizer(lr).minimize(self.pretrain_loss)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        a = a[np.newaxis, :]
        # td = td[np.newaxis, :]
        # a = np.array([a]).reshape(-1, 1)
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # 获取所有操作的概率
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), probs  # return a int

    def pretrain(self, state, action):
        print("Pre-training for Actor!")
        expert_action_batch = np.zeros((self.batch_size, 8))
        for i, a in enumerate(action):
            expert_action_batch[i, a] = 1
        self.sess.run(self.pretrain_op, {self.s: state, self.expert_action: expert_action_batch})

    def contruct_layer(self, inp, activation_fn, reuse, norm, is_train, scope):
        if norm == 'batch_norm':
            out = tf.contrib.layers.batch_norm(inp, activation_fn=activation_fn,
                                               reuse=reuse, is_training=is_train,
                                               scope=scope)
        elif norm == 'None':
            out = activation_fn(inp)
        else:
            ValueError('Can\'t recognize {}'.format(norm))
        return out

    def construct_weights(self):
        weights = {}

        weights['embed_w1'] = tf.Variable(tf.glorot_uniform_initializer()([1, 4]), name='embed_w1')
        weights['embed_b1'] = tf.Variable(tf.zeros([4]), name='embed_b1')

        # for phase, one-hot
        weights['embed_w2'] = tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05)([2, 4]), name='embed_w2')
        #weights['embed_b2'] = tf.Variable(tf.zeros([4]), name='embed_b2')

        # lane embeding
        weights['lane_embed_w3'] = tf.Variable(tf.glorot_uniform_initializer()([8, 16]), name='lane_embed_w3')
        weights['lane_embed_b3'] = tf.Variable(tf.zeros([16]), name='lane_embed_b3')

        # relation embeding, one-hot
        weights['relation_embed_w4'] = tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05)([2, 4]), name='relation_embed_w4')
        #weights['relation_embed_b4'] = tf.Variable(tf.zeros([4]), name='relation_embed_b4')

        weights['feature_conv_w1'] = tf.Variable(tf.glorot_uniform_initializer()([1, 1, 32, 20]), name='feature_conv_w1')
        weights['feature_conv_b1'] = tf.Variable(tf.zeros([20]), name='feature_conv_b1')

        weights['phase_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, 4, 20]), name='phase_conv_w1')
        weights['phase_conv_b1'] = tf.Variable(tf.zeros([20]), name='phase_conv_b1')

        weights['combine_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, 20, 20]), name='combine_conv_w1')
        weights['combine_conv_b1'] = tf.Variable(tf.zeros([20]), name='combine_conv_b1')

        weights['final_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, 20, 1]), name='final_conv_w1')
        weights['final_conv_b1'] = tf.Variable(tf.zeros([1]), name='final_conv_b1')

        return weights

    def construct_forward(self, inp, reuse, norm, is_train, scope, prefix='fc'):
        # embedding, only for 4 or 8 phase, hard code for lane_num_vehicle + cur_phase
        with tf.variable_scope(scope):
            weights = self.construct_weights()
            dim = int(inp.shape[1].value / 2)
            num_veh = inp[:, :dim]
            num_veh = tf.reshape(num_veh, [-1, 1])

            phase = inp[:, dim:]
            phase = tf.cast(phase, tf.int32)
            phase = tf.one_hot(phase, 2)
            phase = tf.reshape(phase, [-1, 2])

            embed_num_veh = self.contruct_layer(tf.matmul(num_veh, weights['embed_w1']) + weights['embed_b1'],
                                     activation_fn=tf.nn.sigmoid, reuse=reuse, is_train=is_train,
                                     norm=norm, scope='num_veh_embed.' + prefix
                                     )
            embed_num_veh = tf.reshape(embed_num_veh, [-1, dim, 4])

            embed_phase = self.contruct_layer(tf.matmul(phase, weights['embed_w2']),
                                     activation_fn=tf.nn.sigmoid, reuse=reuse, is_train=is_train,
                                     norm=norm, scope='phase_embed.' + prefix
                                     )
            embed_phase = tf.reshape(embed_phase, [-1, dim, 4])

            dic_lane = {}
            for i, m in enumerate(self.dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]):
                dic_lane[m] = tf.concat([embed_num_veh[:, i, :], embed_phase[:, i, :]], axis=-1)


            list_phase_pressure = []
            phase_startLane_mapping = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_sameStartLane_mapping"]
            for phase in self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase"]:
                t1 = tf.Variable(tf.zeros(1))
                t2 = tf.Variable(tf.zeros(1))
                for lane in phase_startLane_mapping[phase][0]:
                    t1 += self.contruct_layer(
                       tf.matmul(dic_lane[lane], weights['lane_embed_w3']) + weights['lane_embed_b3'],
                       activation_fn=self._activation_fn, reuse=reuse, is_train=is_train,
                       norm=norm, scope='lane_embed.' + prefix
                       )
                t1 /= len(phase_startLane_mapping[phase][0])

                if len(phase_startLane_mapping[phase]) >= 2:
                    for lane in phase_startLane_mapping[phase][1]:
                        t2 += self.contruct_layer(
                           tf.matmul(dic_lane[lane], weights['lane_embed_w3']) + weights['lane_embed_b3'],
                           activation_fn=self._activation_fn, reuse=reuse, is_train=is_train,
                           norm=norm, scope='lane_embed.' + prefix
                           )
                    t2 /= len(phase_startLane_mapping[phase][1])

                list_phase_pressure.append(t1 + t2)
                # TODO check batch_size here
            constant = relation(self.dic_traffic_env_conf["LANE_PHASE_INFO"])

            constant = tf.one_hot(constant, 2)
            s1, s2 = constant.shape[1:3]
            constant = tf.reshape(constant, (-1, 2))
            relation_embedding = tf.matmul(constant, weights['relation_embed_w4'])
            relation_embedding = tf.reshape(relation_embedding, (-1, s1, s2, 4))

            list_phase_pressure_recomb = []
            num_phase = len(list_phase_pressure)

            for i in range(num_phase):
                for j in range(num_phase):
                    if i != j:
                        list_phase_pressure_recomb.append(
                            tf.concat([list_phase_pressure[i], list_phase_pressure[j]], axis=-1,
                                        name="concat_compete_phase_%d_%d" % (i, j)))

            list_phase_pressure_recomb = tf.concat(list_phase_pressure_recomb, axis=-1 , name="concat_all")
            feature_map = tf.reshape(list_phase_pressure_recomb, (-1, num_phase, num_phase-1, 32))
            #if num_phase == 8:
            #    feature_map = tf.reshape(list_phase_pressure_recomb, (-1, 8, 7, 32))
            #else:
            #    feature_map = tf.reshape(list_phase_pressure_recomb, (-1, 4, 3, 32))

            lane_conv = tf.nn.conv2d(feature_map, weights['feature_conv_w1'], [1, 1, 1, 1], 'VALID', name='feature_conv') + weights['feature_conv_b1']
            lane_conv = tf.nn.leaky_relu(lane_conv, name='feature_activation')


            # relation conv layer
            relation_conv = tf.nn.conv2d(relation_embedding, weights['phase_conv_w1'], [1, 1, 1, 1], 'VALID',
                                     name='phase_conv') + weights['phase_conv_b1']
            relation_conv = tf.nn.leaky_relu(relation_conv, name='phase_activation')
            combine_feature = tf.multiply(lane_conv, relation_conv, name="combine_feature")

            # second conv layer
            hidden_layer = tf.nn.conv2d(combine_feature, weights['combine_conv_w1'], [1, 1, 1, 1], 'VALID', name='combine_conv') + \
                        weights['combine_conv_b1']
            hidden_layer = tf.nn.leaky_relu(hidden_layer, name='combine_activation')

            before_merge = tf.nn.conv2d(hidden_layer, weights['final_conv_w1'], [1, 1, 1, 1], 'VALID',
                                        name='final_conv') + \
                           weights['final_conv_b1']

            #if self.num_actions == 8:
            #    _shape = (-1, 8, 7)
            #else:
            #    _shape = (-1, 4, 3)
            _shape = (-1, 8, 7)
            before_merge = tf.reshape(before_merge, _shape)
            out = tf.reduce_sum(before_merge, axis=2)
            out = tf.nn.softmax(out)

        return out


