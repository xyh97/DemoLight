from learner import Learner
from sampler import BatchSampler
import config
import time
import copy
from multiprocessing import Process
import pickle
import random
import numpy as np
import tensorflow as tf
from script import parse, choose_traffic_file, config_all, parse_roadnet, write_summary
import os
from config import DIC_ENVS
from keras.layers import Input, Dense, Flatten, Reshape, Layer, Lambda, RepeatVector, Activation, Embedding, Conv2D
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers.merge import concatenate, add, dot, maximum, multiply
from keras import backend as K
from keras.initializers import RandomNormal, Constant
from baseline.sotl_agent import SOTLAgent
from actor import Actor
# from ad_critic import Critic
# from q_critic import Critic
# from vq_critic import Critic
from critic_dqfd import Critic
from actor_critic import Agent
from Config import Config, DQfDConfig
from collections import deque
import itertools


def main(args):
    traffic_file_list = choose_traffic_file(args)

    process_list = []
    for traffic_file in traffic_file_list:

        traffic_of_tasks = [traffic_file]

        ### *** exp, agent, traffic_env, path_conf
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = config_all(args)
        # path
        _time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
        postfix = ""
        dic_path.update({
            "PATH_TO_MODEL": os.path.join(dic_path["PATH_TO_MODEL"], traffic_file + "_" + _time + postfix),
            "PATH_TO_WORK_DIRECTORY": os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"],
                                                   traffic_file + "_" + _time + postfix),
            "PATH_TO_GRADIENT": os.path.join(dic_path["PATH_TO_GRADIENT"], traffic_file + "_" + _time + postfix,
                                             "gradient"),
            # "PATH_TO_DATA": os.path.join(dic_path["PATH_TO_DATA"], traffic_file.split(".")[0])
        })
        # traffic env
        dic_traffic_env_conf["TRAFFIC_FILE"] = traffic_file
        dic_traffic_env_conf["TRAFFIC_IN_TASKS"] = [traffic_file]
        if ".json" in traffic_file:
            dic_traffic_env_conf.update({"SIMULATOR_TYPE": "anon"})
        elif ".xml" in traffic_file:
            dic_traffic_env_conf.update({"SIMULATOR_TYPE": "sumo"})
        else:
            raise (ValueError)
        # parse roadnet
        roadnet_path = os.path.join(dic_path['PATH_TO_DATA'], dic_traffic_env_conf['ROADNET_FILE'])
        lane_phase_info = parse_roadnet(roadnet_path)
        dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
        dic_traffic_env_conf["num_lanes"] = int(len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4) # num_lanes per direction
        dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

        dic_exp_conf.update({
            "TRAFFIC_FILE": traffic_file,  # Todo
            "TRAFFIC_IN_TASKS": traffic_of_tasks})

        single_process = False
        if single_process:
            _train(copy.deepcopy(dic_exp_conf),
                   copy.deepcopy(dic_agent_conf),
                   copy.deepcopy(dic_traffic_env_conf),
                   copy.deepcopy(dic_path))
        else:
            p = Process(target=_train, args=(copy.deepcopy(dic_exp_conf),
                                             copy.deepcopy(dic_agent_conf),
                                             copy.deepcopy(dic_traffic_env_conf),
                                             copy.deepcopy(dic_path)))

            process_list.append(p)

    num_process = args.num_process
    if not single_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < num_process:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < num_process:
                continue

            idle = check_all_workers_working(list_cur_p)

            while idle == -1:
                time.sleep(1)
                idle = check_all_workers_working(
                    list_cur_p)
            del list_cur_p[idle]

        for i in range(len(list_cur_p)):
            p = list_cur_p[i]
            p.join()


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def convert_to_input(state, dic_traffic_env_conf):
    inputs = []
    all_start_lane = dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]
    phase = [0] * len(all_start_lane)
    inputs.extend(state['lane_num_vehicle'])
    start_lane = dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_startLane_mapping"][state["cur_phase"][0]]
    for lane in start_lane:
        phase[all_start_lane.index(lane)] = 1
    inputs.extend(phase)
    inputs = np.array(inputs)

    return inputs

def set_n_step(container, n):
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    n_step_reward = sum([t[2] * 0.8 **i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + 10 - 1)
        n_step_reward += t_list[end][2]*0.8**(end-begin)
        # extend[n_reward, n_next_s, n_done, actual_n]
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/0.8
    return t_list


def _train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    random.seed(dic_agent_conf['SEED'])
    np.random.seed(dic_agent_conf['SEED'])
    tf.set_random_seed(dic_agent_conf['SEED'])



    dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], "train_round")
    #
    # dic_path['PATH_TO_SUMO_CONF'] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], "sumo_conf", task)

    if not os.path.exists(dic_path['PATH_TO_LOG']):
        os.makedirs(dic_path['PATH_TO_LOG'])

    # dic_exp_conf = copy.deepcopy(self.dic_exp_conf)

    if dic_traffic_env_conf['SIMULATOR_TYPE'] == 'sumo':
        path_to_work_directory = dic_path["PATH_TO_SUMO_CONF"]
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=dic_path["PATH_TO_LOG"],
            path_to_work_directory=path_to_work_directory,
            dic_traffic_env_conf=dic_traffic_env_conf)

    elif dic_traffic_env_conf['SIMULATOR_TYPE'] == 'anon':
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=dic_path["PATH_TO_LOG"],
            path_to_work_directory=dic_path["PATH_TO_DATA"],
            dic_traffic_env_conf=dic_traffic_env_conf)
    dic_agent_conf["PHI"] = 5
    dic_agent_conf["MIN_GREEN_VEC"] = 3
    dic_agent_conf["MAX_RED_VEC"] = 6
    demo_path = "../frap/demo_{}.p".format(dic_exp_conf["TRAFFIC_FILE"])

    with open(demo_path, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))

    sess = tf.InteractiveSession()
    actor = Actor(sess=sess, n_features=16, n_actions=8, dic_traffic_env_conf=dic_traffic_env_conf, lr=1e-3)  # 初始化Actor
    critic = Critic(sess=sess, n_features=16, config=DQfDConfig(), dic_traffic_env_conf=dic_traffic_env_conf,
                    demo=demo_transitions, lr=1e-3)  # 初始化Critic
    # agent = Agent(sess=sess, n_features=16, config=DQfDConfig(), dic_traffic_env_conf=dic_traffic_env_conf,
    #               demo=demo_transitions, lr=1e-3)
    # actor = Actor(sess=sess, n_features=16, n_actions=8, dic_traffic_env_conf=dic_traffic_env_conf, lr=1e-3)
    # critic = Critic(sess=sess, n_features=16, lr=1e-3)
    sess.run(tf.global_variables_initializer())  # 初始化参数

    for i in range(10):
        state, action = critic.train_Q_network(pre_train=True)
        actor.pretrain(state, action)

    for i in range(501):

        done = False
        state = env.reset()
        step_num = 0
        while not done and step_num < int(dic_exp_conf["EPISODE_LEN"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for one_state in state:
                s = convert_to_input(one_state, dic_traffic_env_conf)
                action, probs = actor.choose_action(s)  # one for multi-state, the other for multi-intersection
                # action, probs = agent.choose_action(s)
                action_list.append(action)  # for multi-state

            next_state, reward, done, ave_reward = env.step(action_list)

            s = convert_to_input(state[0], dic_traffic_env_conf)
            s_ = convert_to_input(next_state[0], dic_traffic_env_conf)

            next_action, _ = actor.choose_action(s_)
            # next_action, _ = agent.choose_action(s_)
            # #
            # q_a = critic.learn(s, np.array(ave_reward), s_, np.array([action]), np.array([next_action]), probs)
            # q_a = critic.learn(s, np.array(ave_reward), s_, np.array([action]), np.array([next_action]))
            if i != 0:
                q_a = critic.learn(s, np.array(ave_reward), s_, np.array([action]), np.array([next_action]), probs)
            # q_a = critic.learn(s, np.array(reward), s_)
                actor.learn(s, np.array([action]), q_a)
            # agent.learn_actor(s, np.array([action]), q_a)

            state = next_state
            step_num += 1
        # if i % 3 == 0 and i != 0:
        #     critic.sess.run(critic.update_target_net)
        env.bulk_log(i)
        write_summary(dic_path, dic_exp_conf, i)

    # with open(Config.DEMO_DATA_PATH, 'rb') as f:
    #     demo_transitions = pickle.load(f)
    #     demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
    #     assert len(demo_transitions) == Config.demo_buffer_size
    #
    # actor = Actor(sess=sess, n_features=16, n_actions=8, lr=1e-3)
    # critic = Critic(sess=sess, n_features=16, lr=1e-3)
    #
    # e, replay_full_episode = 0, None
    # while True:
    #     done = False
    #     step_num = 0
    #     state = env.reset()
    #     t_q = deque(maxlen=Config.trajectory_n)
    #     n_step_reward = None
    #     while not done and step_num < int(dic_exp_conf["EPISODE_LEN"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
    #         action_list = []
    #         for one_state in state:
    #             s = convert_to_input(one_state, dic_traffic_env_conf)
    #             action, probs = actor.choose_action(s)  # one for multi-state, the other for multi-intersection
    #             action_list.append(action)  # for multi-state
    #
    #         next_state, _, done, ave_reward = env.step(action_list)
    #         s = convert_to_input(state[0], dic_traffic_env_conf)
    #         s_ = convert_to_input(next_state[0], dic_traffic_env_conf)
    #         next_action, _ = actor.choose_action(s_)
    #         reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]
    #         t_q.append([s, action, probs[0, action], ave_reward[0], s_, done, 0.0])
    #         if len(t_q) == t_q.maxlen:
    #             if n_step_reward is None:
    #                 n_step_reward = sum([t[3] * Config.GAMMA ** i for i, t in enumerate(t_q)])
    #             else:
    #                 n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
    #                 n_step_reward += ave_reward[0] * Config.GAMMA ** (Config.trajectory_n - 1)
    #             t_q[0].extend([n_step_reward, s_, done, t_q.maxlen])
    #             critic.perceive(t_q[0])
    #             if critic.replay_memory.full():
    #                 state_batch, action_batch, td_error_batch = critic.train_Q_network(actor=actor, update=False)
    #                 actor.learn(state_batch, action_batch, td_error_batch)
    #                 replay_full_episode = replay_full_episode or e
    #         state = next_state
    #
    #     if done:
    #         t_q.popleft()
    #         transitions = set_n_step(t_q, Config.trajectory_n)
    #         for t in transitions:
    #             critic.perceive(t)
    #             if critic.replay_memory.full():
    #                 state_batch, action_batch, td_error_batch = critic.train_Q_network(actor=actor, update=False)
    #                 actor.learn(state_batch, action_batch, td_error_batch)
    #                 replay_full_episode = replay_full_episode or e
    #
    #         if critic.replay_memory.full():
    #             critic.sess.run(critic.update_target_net)
    #
    #         print("episode: {}  memory length: {}"
    #               .format(e, len(critic.replay_memory)))
    #         env.bulk_log(e)
    #         write_summary(dic_path, dic_exp_conf, e)
    #
    #     if e >= Config.episode:
    #         break
    #
    #     e += 1



if __name__ == '__main__':
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    main(args)
