import sys
sys.path.append("..")
from baseline.agent import Agent
import random


class SOTLAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        super(SOTLAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path)

        self.current_phase_time = 0

        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == "anon":
            self.DIC_PHASE_MAP = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                0: 0
            }
            self.green_4_lane = {
                0: [0, 1],
                1: [2, 3],
            }
            self.green_8_lane = {
                0: [1, 3],
                1: [5, 7],
                2: [0, 2],
                3: [4, 6],
                4: [0, 1],
                5: [2, 3],
                6: [6, 7],
                7: [4, 5],
            }

        else:
            self.DIC_PHASE_MAP = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                -1: -1
            }


    def choose_action(self, state):
        ''' choose the best action for current state '''

        if state["cur_phase"][0] == -1:
            return self.action
        cur_phase = self.DIC_PHASE_MAP[state["cur_phase"][0]]
        all_start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]
        start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_startLane_mapping"][state["cur_phase"][0]]
        indexes = []
        for lane in start_lane:
            indexes.append(all_start_lane.index(lane))

        #print(state)
        #print(state["time_this_phase"][0], self.dic_agent_conf["PHI"], cur_phase)
        # if len(self.dic_traffic_env_conf["PHASE"]) == 2:
        #     green_lane = self.green_4_lane
        # else:
        #     green_lane = self.green_8_lane

        if state["time_this_phase"][0] >= self.dic_agent_conf["PHI"]:
            green_vec = sum([state["lane_num_vehicle_been_stopped_thres1"][i] for i in indexes])
            red_vec = sum(state["lane_num_vehicle_been_stopped_thres1"]) - green_vec
            print("green: %d, red: %d"%(green_vec, red_vec))
            if green_vec <= self.dic_agent_conf["MIN_GREEN_VEC"] and \
                red_vec > self.dic_agent_conf["MAX_RED_VEC"]:
                self.current_phase_time = 0
                # self.action = (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"])
                # self.action = self.next_phase[cur_phase]
                return (cur_phase + 1) % 8
                # return self.action
            else:
                self.action = cur_phase
                self.current_phase_time += 1
                return cur_phase
        else:
            self.action = cur_phase
            self.current_phase_time += 1
            return cur_phase

