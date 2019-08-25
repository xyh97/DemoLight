import tensorflow as tf
import numpy as np
import functools
from Memory import Memory
import random


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


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


class Agent():
    def __init__(self, sess, n_features, config, dic_traffic_env_conf, demo=None, lr=0.01):
        self.sess = sess
        self.config = config
        self._activation_fn = tf.nn.leaky_relu
        self.dic_traffic_env_conf = dic_traffic_env_conf

        # replay buffer
        self.replay_memory = Memory(capacity=self.config.replay_buffer_size, permanent_data=len(demo))
        # self.replay_memory = None
        self.demo_memory = Memory(capacity=self.config.demo_buffer_size, permanent_data=self.config.demo_buffer_size)
        self.add_demo_to_memory(demo_transitions=demo)
        self.state_dim = 16
        self.action_dim = 8

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.q_a_ = tf.placeholder(tf.float32, [None, 1], "q_next")
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a = tf.placeholder(tf.int32, [None, 1], 'act')
        self.act_probs = tf.placeholder(tf.float32, [None, 8], 'act_probs')

        self.action_batch = tf.placeholder("int32", [None])
        self.y_input = tf.placeholder("float", [None, self.action_dim])
        self.ISWeights = tf.placeholder("float", [None, 1])
        self.n_step_y_input = tf.placeholder("float", [None, self.action_dim])  # for n-step reward
        self.isdemo = tf.placeholder("float", [None])

        self.td = tf.placeholder(tf.float32, [None, 1], "td_error")  # TD_error
        self.expert_action = tf.placeholder(tf.float32, [None, 8], "expert_action")

        self.hidden = self.construct_forward(self.s, True, 'None', True, "hidden", prefix='fc')

        with tf.variable_scope('Q-Value'):
            self.q = tf.layers.dense(
                inputs=self.hidden,
                units=8,  # number of hidden units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Q'
            )

        with tf.variable_scope('Q-Target'):
            self.q_target = tf.layers.dense(
                inputs=self.hidden,
                units=8,  # number of hidden units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Q-Target'
            )

        with tf.variable_scope('Actor'):
            self.probs = tf.layers.dense(
                inputs=self.hidden,
                units=8,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        # self.v = self.build_net("Value")
        # self.q = self.construct_forward(self.s, True, 'None', True, "Q-Value", prefix='fc')
        # self.q_target = self.construct_forward(self.s, True, 'None', True, "Q-Target", prefix='fc')
        # self.q = self.build_q_net("Q-Value")
        # self.q_target = self.build_q_net("Q-Target")
        self.q_a = tf.batch_gather(self.q, self.a)
        self.v = tf.reduce_sum(self.q * self.act_probs, axis=1, keepdims=True)
        # self.v_target = self.build_net("Target")
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q-Target')
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q-Value')
        self.replace_target_op = [tf.assign(t, p) for t, p in zip(self.t_params, self.params)]

        self.loss
        self.optimize
        self.update_target_net
        self.abs_errors
        self.time_step = 0

        with tf.variable_scope('squared_TD_error'):
            # self.td_error = self.r + 0.8 * self.v_ - self.v
            self.td_error = self.q_a - self.v
            q_loss = tf.reduce_mean(tf.squared_difference(self.q_a, self.r + 0.8 * self.q_a_))
            # v_loss = tf.reduce_mean(tf.squared_difference(self.v, self.r + 0.8 * self.v_))
            # self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
            # self.loss = q_loss + v_loss
            self.los = q_loss
        with tf.variable_scope('train-c'):
            self.train_op_critic = tf.train.AdamOptimizer(lr).minimize(self.los)

        with tf.variable_scope('exp_v'):
            # log_prob = tf.log(self.acts_prob[0, self.a])
            log_prob = tf.log(tf.batch_gather(self.probs, self.a))
            self.exp_v = tf.reduce_mean(log_prob * self.td)  # advantage (TD_error) guided loss

        self.action = gumbel_softmax(logits=self.probs, temperature=1, hard=False)

        with tf.variable_scope('train-a'):
            self.train_op_actor = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

        self.pretrain_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.action, labels=self.expert_action)
        self.pretrain_op = tf.train.AdamOptimizer(lr).minimize(self.pretrain_loss)

    def add_demo_to_memory(self, demo_transitions):
        # add demo data to both demo_memory & replay_memory
        for t in demo_transitions:
            self.demo_memory.store(np.array(t, dtype=object))
            self.replay_memory.store(np.array(t, dtype=object))
            assert len(t) == 10

    # use the expert-demo-data to pretrain
    def pre_train(self):
        print('Pre-training ...')
        for i in range(self.config.PRETRAIN_STEPS):
            self.train_Q_network(pre_train=True)
            if i % 200 == 0 and i > 0:
                print('{} th step of pre-train finish ...'.format(i))
        self.time_step = 0
        print('All pre-train finish.')

    @lazy_property
    def abs_errors(self):
        return tf.reduce_sum(tf.abs(self.y_input - self.q), axis=1)  # only use 1-step R to compute abs_errors

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
        return optimizer.minimize(self.loss)  # only parameters in select-net is optimized here

    @lazy_property
    def update_target_net(self):
        select_params = tf.get_collection('Q-Value')
        eval_params = tf.get_collection('Q-Target')
        return [tf.assign(e, s) for e, s in zip(eval_params, select_params)]

    def learn_critic(self, s, r, s_, a, next_a, probs):
        s, s_, r = s[np.newaxis, :], s_[np.newaxis, :], r[np.newaxis, :]
        a, next_a = a[np.newaxis, :], next_a[np.newaxis, :]

        # v_ = self.sess.run(self.v, {self.s: s_})
        q_a_ = self.sess.run(self.q_a, {self.s: s_, self.a: next_a})
        td_error, _ = self.sess.run([self.td_error, self.train_op_critic],
                                    {self.s: s,  self.r: r, self.act_probs: probs,
                                     self.q_a_: q_a_, self.a: a})
        return td_error

    def loss_l(self, ae, a):
        return 0.0 if ae == a else 0.8

    def loss_jeq(self, q):
        jeq = 0.0
        for i in range(self.config.BATCH_SIZE):
            ae = self.action_batch[i]
            max_value = float("-inf")
            for a in range(self.action_dim):
                max_value = tf.maximum(q[i][a] + self.loss_l(ae, a), max_value)
            jeq += self.isdemo[i] * (max_value - q[i][ae])
        return jeq

    @lazy_property
    def loss(self):
        l_dq = tf.reduce_mean(tf.squared_difference(self.q, self.y_input))
        l_n_dq = tf.reduce_mean(tf.squared_difference(self.q, self.n_step_y_input))
        l_jeq = self.loss_jeq(self.q)
        l_l2 = tf.reduce_sum([tf.reduce_mean(reg_l) for reg_l in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
        return self.ISWeights * tf.reduce_sum([l * λ for l, λ in zip([l_dq, l_n_dq, l_jeq, l_l2], self.config.LAMBDA)])

    def train_Q_network(self, pre_train=False, update=True):
        """
        :param pre_train: True means should sample from demo_buffer instead of replay_buffer
        :param update: True means the action "update_target_net" executes outside, and can be ignored in the function
        """
        if not pre_train and not self.replay_memory.full():  # sampling should be executed AFTER replay_memory filled
            return
        self.time_step += 1

        assert self.replay_memory.full() or pre_train

        actual_memory = self.demo_memory if pre_train else self.replay_memory
        tree_idxes, minibatch, ISWeights = actual_memory.sample(self.config.BATCH_SIZE)

        np.random.shuffle(minibatch)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]
        demo_data = [data[5] for data in minibatch]
        n_step_reward_batch = [data[6] for data in minibatch]
        n_step_state_batch = [data[7] for data in minibatch]
        n_step_done_batch = [data[8] for data in minibatch]
        actual_n = [data[9] for data in minibatch]

        # provide for placeholder，compute first
        q_next = self.q.eval(feed_dict={self.s: next_state_batch})
        q_target_next = self.q_target.eval(feed_dict={self.s: next_state_batch})

        n_step_q_next = self.q.eval(feed_dict={self.s: n_step_state_batch})
        n_step_q_target_next = self.q_target.eval(feed_dict={self.s: n_step_state_batch})

        y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        n_step_y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        # td_error_batch = np.zeros((self.config.BATCH_SIZE, 1))
        for i in range(self.config.BATCH_SIZE):
            # state, action, reward, next_state, done, demo_data, n_step_reward, n_step_state, n_step_done = t
            temp = self.q.eval(feed_dict={self.s: state_batch[i].reshape((-1, self.state_dim))})[0]
            # v = np.sum(temp, action_prob_batch[i])
            # td_error_batch[i] = temp[action_batch[i]] - v
            temp_0 = np.copy(temp)
            # add 1-step reward
            action = np.argmax(q_next[i])
            # action = next_action_batch[i]
            temp[action_batch[i]] = reward_batch[i] + (1 - int(done_batch[i])) * self.config.GAMMA * q_target_next[i][action]
            y_batch[i] = temp
            # add n-step reward
            action = np.argmax(n_step_q_next[i])
            q_n_step = (1 - int(n_step_done_batch[i])) * self.config.GAMMA**actual_n[i] * n_step_q_target_next[i][action]
            temp_0[action_batch[i]] = n_step_reward_batch[i] + q_n_step
            n_step_y_batch[i] = temp_0

        _, abs_errors = self.sess.run([self.optimize, self.abs_errors],
                                      feed_dict={self.y_input: y_batch,
                                                 self.n_step_y_input: n_step_y_batch,
                                                 self.s: state_batch,
                                                 self.action_batch: action_batch,
                                                 self.isdemo: demo_data,
                                                 self.ISWeights: ISWeights})

        self.replay_memory.batch_update(tree_idxes, abs_errors)  # update priorities for data in memory

        # 此例中一局步数有限，因此可以外部控制一局结束后update ，update为false时在外部控制
        if update and self.time_step % self.config.UPDATE_TARGET_NET == 0:
            self.sess.run(self.update_target_net)

        return state_batch, action_batch

    def learn_actor(self, s, a, td):
        s = s[np.newaxis, :]
        a = a[np.newaxis, :]
        # td = td[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td: td}
        _, exp_v = self.sess.run([self.train_op_actor, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.probs, {self.s: s})  # 获取所有操作的概率
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

            before_merge = tf.nn.leaky_relu(before_merge, name='combine_activation')

            #if self.num_actions == 8:
            #    _shape = (-1, 8, 7)
            #else:
            #    _shape = (-1, 4, 3)
            _shape = (-1, 8, 7)
            before_merge = tf.reshape(before_merge, _shape)
            out = tf.reduce_sum(before_merge, axis=2)

        return out