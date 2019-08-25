import multiprocessing as mp
from episode import BatchEpisodes

from config import DIC_ENVS
import json
import os
import shutil
import random
import copy
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from math import isnan
from subproc_vec_env import SubprocVecEnv
from script import write_summary
import pickle

class BatchSampler(object):

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                           dic_path, batch_size, num_workers=2):
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.traffic_in_tasks = dic_exp_conf['TRAFFIC_IN_TASKS']
        # num of episodes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.queue = mp.Queue()

        self.envs = None

        self._task_id = 0
        # do file operations

        # sumo relevant
        self._LIST_SUMO_FILES = [
            "cross.car.type.xml",
            "cross.con.xml",
            "cross.edg.xml",
            "cross.net.xml",
            "cross.netccfg",
            "cross.nod.xml",
            "cross.sumocfg",
            "cross.tll.xml",
            "cross.typ.xml"
        ]

        self._path_check()
        self._copy_conf_file()
        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'sumo':
            self._copy_sumo_file()
            #self._modify_sumo_file()
        #elif self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'anon':
        #    self._copy_anon_file()

        self.path_to_log = self.dic_path['PATH_TO_WORK_DIRECTORY']

        self.step = 0
        self.target_step = 0
        self.lr_step = 0

        self.test_step = 0

    def _path_check(self):
        # check path
        if not os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

        if not os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            os.makedirs(self.dic_path["PATH_TO_MODEL"])

        if not os.path.exists(self.dic_path["PATH_TO_GRADIENT"]):
            os.makedirs(self.dic_path["PATH_TO_GRADIENT"])

        if self.dic_exp_conf["PRETRAIN"]:
            if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
                pass
            else:
                os.makedirs(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

            if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
                pass
            else:
                os.makedirs(self.dic_path["PATH_TO_PRETRAIN_MODEL"])

    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_anon_file(self, path=None):
        # hard code !!!
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files

        for traffic in self.dic_exp_conf["TRAFFIC_IN_TASKS"]:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], traffic),
                            os.path.join(path, traffic))
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["ROADNET_FILE"]),
                        os.path.join(path, self.dic_traffic_env_conf["ROADNET_FILE"]))

    def _modify_sumo_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # modify sumo files
        for traffic in self.dic_exp_conf["TRAFFIC_IN_TASKS"]:
            self._set_traffic_file(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "cross.sumocfg"),
                                   os.path.join(path, traffic, "cross.sumocfg"),
                                   traffic)
        return

    def _copy_sumo_file(self, path=None):

        for traffic in self.dic_exp_conf["TRAFFIC_IN_TASKS"]:
            path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "sumo_conf", traffic)
            if not os.path.exists(path):
                os.makedirs(path)
            # copy sumo files
            for file_name in self._LIST_SUMO_FILES:
                shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], file_name),
                            os.path.join(path, file_name))
            #for file_name in self.dic_exp_conf["TRAFFIC_FILE"]:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], traffic),
                        os.path.join(path, traffic))
            # modify sumo files
            self._set_traffic_file(os.path.join(path, "cross.sumocfg"),
                                   os.path.join(path, "cross.sumocfg"),
                                   [traffic])
        return

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    def sample_period(self, policy, task, batch_id, params=None, target_params=None, old_episodes=None):
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        episodes = BatchEpisodes(batch_size=self.batch_size, dic_traffic_env_conf=self.dic_traffic_env_conf, dic_agent_conf=self.dic_agent_conf, old_episodes=old_episodes)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params: # todo precise load parameter logic
            policy.load_params(params)

        while (not all(dones)) or (not self.queue.empty()):

            if self.dic_traffic_env_conf['MODEL_NAME'] == 'MetaDQN':
                actions = policy.choose_action(observations, task_type=task_type)
            else:
                actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

            # if update
            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf['UPDATE_PERIOD'] == 0:
                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:
                    #TODO
                    episodes.forget()

                policy.fit(episodes, params=params, target_params=target_params)
                params = policy.update_params(episodes, params=copy.deepcopy(params), lr_step=self.lr_step)
                policy.load_params(params)

                self.lr_step += 1
                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:
                    target_params = params
                    self.target_step = 0

            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf['TEST_PERIOD'] == 0:
                self.test(policy, task, self.test_step, params=params)
                pickle.dump(params, open(
                    os.path.join(self.dic_path['PATH_TO_MODEL'], 'params' + "_" + str(self.test_step) + ".pkl"),
                    'wb'))

                self.test_step += 1
            self.step += 1

        return params, target_params, episodes

    def test_sample(self, policy, tasks, batch_id, params=None):
        for i in range(len(tasks)):
            self.queue.put(i)
        for _ in range(len(tasks)):
            self.queue.put(None)

        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params: # todo precise load parameter logic
            policy.load_params(params)

        while (not all(dones)) or (not self.queue.empty()):
            if self.dic_traffic_env_conf['MODEL_NAME'] == 'MetaDQN':
                actions = []
                for i in range(len(tasks)):
                    task = tasks[i]
                    task_type = None
                    for type in self.dic_traffic_env_conf['TASK_TYPE'].keys():
                        traffic_list = self.dic_traffic_env_conf['TASK_TYPE'][type]
                        if task in traffic_list:
                            task_type = type
                    if task_type == None:
                        raise (ValueError)

                    actions.append(policy.choose_action([observations[i]], test=True, task_type=task_type)[0])
            else:
                actions = policy.choose_action(observations, test=True)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            observations, batch_ids = new_observations, new_batch_ids

    def test(self, policy, task, batch_id, params):
        policy.load_params(params)
        task_id = self.dic_traffic_env_conf['TRAFFIC_IN_TASKS'].index(task)

        dic_traffic_env_conf = copy.deepcopy(self.dic_traffic_env_conf)
        dic_traffic_env_conf['TRAFFIC_FILE'] = task

        dic_path = copy.deepcopy(self.dic_path)
        dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], 'test_round',
                                               'task_%d_%s'%(task_id, task), 'tasks_round_' + str(batch_id))

        dic_path['PATH_TO_SUMO_CONF'] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], "sumo_conf", task)

        if not os.path.exists(dic_path['PATH_TO_LOG']):
            os.makedirs(dic_path['PATH_TO_LOG'])

        dic_exp_conf = copy.deepcopy(self.dic_exp_conf)

        if self.dic_traffic_env_conf['SIMULATOR_TYPE'] == 'sumo':
            path_to_work_directory = self.dic_path["PATH_TO_SUMO_CONF"]
            env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                path_to_log=dic_path["PATH_TO_LOG"],
                path_to_work_directory=path_to_work_directory,
                dic_traffic_env_conf=dic_traffic_env_conf)

        elif self.dic_traffic_env_conf['SIMULATOR_TYPE'] == 'anon':
            env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                path_to_log=dic_path["PATH_TO_LOG"],
                path_to_work_directory=dic_path["PATH_TO_DATA"],
                dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while not done and step_num < int(
                dic_exp_conf["EPISODE_LEN"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for one_state in state:
                action = policy.choose_action([[one_state]], test=True) # one for multi-state, the other for multi-intersection
                action_list.append(action[0]) # for multi-state

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            step_num += 1
            stop_cnt += 1
        env.bulk_log()
        write_summary(dic_path, 'task_%d_%s'%(task_id, task), self.dic_exp_conf["EPISODE_LEN"], batch_id)
        return

    def reset_task(self, tasks, batch_id, reset_type='learning'):
        # regenerate new envs to avoid the engine stuck bug!

        #for i in range(self.num_workers):
        dic_agent_conf_list = []
        dic_traffic_env_conf_list = []
        dic_path_list = []
        for task in tasks:
            task_id = self.dic_traffic_env_conf['TRAFFIC_IN_TASKS'].index(task)
            dic_agent_conf = copy.deepcopy(self.dic_agent_conf)
            dic_agent_conf['TRAFFIC_FILE'] = task
            dic_agent_conf_list.append(dic_agent_conf)

            dic_traffic_env_conf = copy.deepcopy(self.dic_traffic_env_conf)
            dic_traffic_env_conf['TRAFFIC_FILE'] = task
            dic_traffic_env_conf_list.append(dic_traffic_env_conf)

            dic_path = copy.deepcopy(self.dic_path)
            if reset_type == 'test':
                dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], reset_type + '_round',
                                                       'task_%d_%s' % (task_id, task), 'tasks_round_' + str(batch_id))
            else:
                dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], reset_type+'_round',
                                                   'tasks_round_' + str(batch_id), 'task_%d_%s'%(task_id, task))
            dic_path['PATH_TO_SUMO_CONF'] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], "sumo_conf", task)
            dic_path_list.append(dic_path)

            if not os.path.exists(dic_path['PATH_TO_LOG']):
                os.makedirs(dic_path['PATH_TO_LOG'])

        self.envs = SubprocVecEnv(dic_path_list, dic_traffic_env_conf_list, len(tasks), queue=self.queue)

