import numpy as np
import copy
import  config

class BatchEpisodes(object):
    def __init__(self, batch_size, dic_traffic_env_conf, dic_agent_conf, gamma=0.95, device='cpu', old_episodes=None):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.all_start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]

        self.total_samples = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self.tot_x = []
        self.tot_next_x = []
        if old_episodes:
            self.total_samples = self.total_samples + old_episodes.total_samples
            self.tot_x = self.tot_x + old_episodes.tot_x
            self.tot_next_x = self.tot_next_x + old_episodes.tot_next_x

    def append(self, observations, actions, new_observations, rewards, batch_ids):
        for observation, action, new_observation, reward, batch_id in zip(
                observations, actions, new_observations, rewards, batch_ids):
            if batch_id is None:
                continue

            self.total_samples.append([observation, action, new_observation, reward, 0, 0])

            # change to fixed cur_phase and lane_num_vehicle
            phase = [0] * len(self.all_start_lane)
            start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_startLane_mapping"][observation[0]["cur_phase"][0]]
            for lane in start_lane:
                phase[self.all_start_lane.index(lane)] = 1
            self.tot_x.append(observation[0]['lane_num_vehicle'] + phase)

            phase = [0] * len(self.all_start_lane)
            start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_startLane_mapping"][
                new_observation[0]["cur_phase"][0]]
            for lane in start_lane:
                phase[self.all_start_lane.index(lane)] = 1
            self.tot_next_x.append(new_observation[0]['lane_num_vehicle'] + phase)

    def get_x(self):
        return np.reshape(np.array(self.tot_x), (len(self.tot_x), -1))

    def get_next_x(self):
        return np.reshape(np.array(self.tot_next_x), (len(self.tot_next_x), -1))

    def forget(self):
        self.total_samples = self.total_samples[-1 * self.dic_agent_conf['MAX_MEMORY_LEN'] : ]
        self.tot_x = self.tot_x[-1 * self.dic_agent_conf['MAX_MEMORY_LEN'] : ]
        self.tot_next_x = self.tot_next_x[-1 * self.dic_agent_conf['MAX_MEMORY_LEN']:]

    def prepare_y(self, q_values):
        self.tot_y = q_values

    def get_y(self):
        return self.tot_y

    def __len__(self):
        return len(self.total_samples)
