# collect the common function
import os
import numpy as np
import pandas as pd
from math import isnan
import config
from traffic import *
import copy
import json
import xml.etree.ElementTree as ET

# get traffic in one lane
def get_traffic_volume(traffic_file):
    # only support "cross" and "synthetic"
    if "cross" in traffic_file:
        sta = traffic_file.find("equal_") + len("equal_")
        end = traffic_file.find(".xml")
        return int(traffic_file[sta:end])

    elif "synthetic" in traffic_file:
        traffic_file_list = traffic_file.split("-")
        volume_list = []
        for i in range(2, 6):
            volume_list.append(int(traffic_file_list[i][2:]))

        vol = min(max(volume_list[0:2]), max(volume_list[2:]))

        return int(vol/100)*100
    elif "flow" in traffic_file:
        sta = traffic_file.find("flow_1_1_") + len("flow_1_1_")
        end = traffic_file.find(".json")
        return int(traffic_file[sta:end])

    elif "real" in traffic_file:
        sta = traffic_file.rfind("-") + 1
        end = traffic_file.rfind(".json")
        return int(traffic_file[sta:end])

    elif "hangzhou" in traffic_file:
        traffic = traffic_file.split(".json")[0]
        vol = int(traffic.split("_")[-1])
        return vol

    elif "ngsim" in traffic_file:
        traffic = traffic_file.split(".json")[0]
        vol = int(traffic.split("_")[-1])
        return vol

## get total number of vehicles
## not very comprehensive

def get_total_traffic_volume(traffic_file):
    # only support "cross" and "synthetic"
    if "cross" in traffic_file:
        sta = traffic_file.find("equal_") + len("equal_")
        end = traffic_file.find(".xml")
        return int(traffic_file[sta:end]) * 4

    elif "synthetic" in traffic_file:
        sta = traffic_file.rfind("-") + 1
        end = traffic_file.rfind(".json")
        return int(traffic_file[sta:end])

    elif "flow" in traffic_file:
        sta = traffic_file.find("flow_1_1_") + len("flow_1_1_")
        end = traffic_file.find(".json")
        return int(traffic_file[sta:end]) * 4

    elif "real" in traffic_file:
        sta = traffic_file.rfind("-") + 1
        end = traffic_file.rfind(".json")
        return int(traffic_file[sta:end])

    elif "hangzhou" in traffic_file:
        traffic = traffic_file.split(".json")[0]
        vol = int(traffic.split("_")[-1])
        return vol
    elif "ngsim" in traffic_file:
        traffic = traffic_file.split(".json")[0]
        vol = int(traffic.split("_")[-1])
        return vol

def write_summary(dic_path, dic_exp_conf, cnt_round):

    episode_len = dic_exp_conf["EPISODE_LEN"]
    traffic_file = dic_exp_conf["TRAFFIC_FILE"]
    record_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_results.csv")
    path_to_seg_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_seg_results.csv")
    num_seg = episode_len // 3600

    if not os.path.exists(path_to_log):
        df_col = pd.DataFrame(columns=("round", "duration", "vec_in", "vec_out"))
        if num_seg > 1:
            list_seg_col = ["round"]
            for i in range(num_seg):
                list_seg_col.append("duration-" + str(i))
            df_seg_col = pd.DataFrame(columns=list_seg_col)
            df_seg_col.to_csv(path_to_seg_log, mode="a", index=False)
        df_col.to_csv(path_to_log, mode="a", index=False)

    # summary items (duration) from csv
    df_vehicle_inter_0 = pd.read_csv(os.path.join(record_dir, "vehicle_inter_0_round_{}.csv".format(cnt_round)),
                                     sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                     names=["vehicle_id", "enter_time", "leave_time"])

    vehicle_in = sum([int(x) for x in (df_vehicle_inter_0["enter_time"].values > 0)])
    vehicle_out = sum([int(x) for x in (df_vehicle_inter_0["leave_time"].values > 0)])
    # duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
    # ave_duration = np.mean([time for time in duration if not isnan(time)])
    # ********* new calculation of duration **********
    df_vehicle_planed_enter = get_planed_entering(os.path.join(dic_path["PATH_TO_DATA"], traffic_file), episode_len)
    ave_duration = cal_travel_time(df_vehicle_inter_0, df_vehicle_planed_enter, episode_len)

    summary = {"round": [cnt_round], "duration": [ave_duration], "vec_in": [vehicle_in], "vec_out": [vehicle_out]}

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(path_to_log, mode="a", header=False, index=False)

    if num_seg > 1:
        list_duration_seg = [float('inf')] * num_seg
        nan_thres = 120
        for i, interval in enumerate(range(0, episode_len, 3600)):
            did = np.bitwise_and(df_vehicle_inter_0["enter_time"].values < interval + 3600,
                                 df_vehicle_inter_0["enter_time"].values > interval)
            duration_seg = df_vehicle_inter_0["leave_time"][did].values - df_vehicle_inter_0["enter_time"][
                did].values
            ave_duration_seg = np.mean([time for time in duration_seg if not isnan(time)])
            # print(traffic_file, round, i, ave_duration)
            real_traffic_vol_seg = 0
            nan_num_seg = 0
            for time in duration_seg:
                if not isnan(time):
                    real_traffic_vol_seg += 1
                else:
                    nan_num_seg += 1

            if nan_num_seg < nan_thres:
                list_duration_seg[i] = ave_duration_seg

        round_summary = {"round": [cnt_round]}
        for j in range(num_seg):
            key = "duration-" + str(j)
            if key not in round_summary.keys():
                round_summary[key] = [list_duration_seg[j]]
        round_summary = pd.DataFrame(round_summary)
        round_summary.to_csv(path_to_seg_log, mode="a", index=False, header=False)

def get_planed_entering(flowFile, episode_len):
    # todo--check with huichu about how each vehicle is inserted, according to the interval. 1s error may occur.
    if 'json' in flowFile:
        list_flow = json.load(open(flowFile, "r"))
        dic_traj = {'vehicle_id':[], 'planed_enter_time':[]}
        for flow_id, flow in enumerate(list_flow):
            list_ts_this_flow = []
            for step in range(flow["startTime"], min(flow["endTime"] + 1, episode_len)):
                if step == flow["startTime"]:
                    list_ts_this_flow.append(step)
                elif step - list_ts_this_flow[-1] >= flow["interval"]:
                    list_ts_this_flow.append(step)

            for vec_id, ts in enumerate(list_ts_this_flow):
                dic_traj['vehicle_id'].append("flow_{0}_{1}".format(flow_id, vec_id))
                dic_traj['planed_enter_time'].append(ts)

        df = pd.DataFrame(dic_traj)
    else:
        tree = ET.parse(flowFile)
        root = tree.getroot()
        vehicle = root.findall('vehicle')
        dic_traj = {'vehicle_id': [], 'planed_enter_time': []}
        for id, v in enumerate(vehicle):
            dic_traj['vehicle_id'].append("{}".format(id))
            dic_traj['planed_enter_time'].append(int(v.attrib['depart']))
        df = pd.DataFrame(dic_traj)
    return df


def cal_travel_time(df_vehicle_actual_enter_leave, df_vehicle_planed_enter, episode_len):
    df_vehicle_planed_enter.set_index('vehicle_id', inplace=True)
    df_vehicle_actual_enter_leave.set_index('vehicle_id', inplace=True)
    df_res = pd.concat([df_vehicle_planed_enter, df_vehicle_actual_enter_leave], axis=1, sort=False)
    assert len(df_res) == len(df_vehicle_planed_enter)

    df_res["leave_time"].fillna(episode_len, inplace=True)
    df_res["travel_time"] = df_res["leave_time"] - df_res["planed_enter_time"]
    travel_time = df_res["travel_time"].mean()
    return travel_time


def parse():
    import argparse
    parser = argparse.ArgumentParser(description='Meta RLSignal')

    parser.add_argument("--memo", type=str, default="default")
    parser.add_argument("--algorithm", type=str, default="TransferDQN")
    parser.add_argument("--traffic", type=str, default='debug')

    parser.add_argument("--pre_train_model_name", type=str, default='random')

    parser.add_argument("--roadnet", type=str, default="roadnet_1_1.json")
    parser.add_argument("--flow_file", type=str, default="flow.json")

    # running time
    parser.add_argument("--episode_len", type=int, default=3600)
    parser.add_argument("--test_episode_len", type=int, default=3600)
    parser.add_argument("--run_round", type=int, default=40)

    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--update_start", type=int, default=100)
    parser.add_argument("--update_period", type=int, default=1)
    parser.add_argument("--test_period", type=int, default=50)

    # process relevant
    parser.add_argument("--num_process", type=int, default=10, help="number of traffic")
    parser.add_argument('--num_generator', type=int, default=1,
                        help='total number of generator')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='maximum number of generator at the same time; <= num_generator')

    # learning rate
    parser.add_argument("--alpha", type=float, default=0.001, help='learning_rate')
    parser.add_argument("--learning_rate_decay_step", type=int, default=100)
    parser.add_argument("--min_alpha", type=float, default=0.001)

    # epsilon
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--min_epsilon", type=float, default=0.2)

    parser.add_argument('--epoch', type=int, default=1,
                        help='number of gradient step when updating para')
    parser.add_argument("--clip_size", type=float, default=1)

    parser.add_argument("--seed", type=int, default=11)

    # rarely change
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sumo_gui", action="store_true")
    parser.add_argument("--done", action="store_true")
    parser.add_argument("--visible_gpu", type=str, default="")

    args = parser.parse_args()
    return args

def choose_traffic_file(args):
    if args.traffic == 'uniform':
        traffic_file_list = uniform_traffic_list
    elif args.traffic == 'large':
        traffic_file_list = large_traffic_list
    elif args.traffic == 'new':
        traffic_file_list = new_traffic_list
    elif args.traffic == 'small':
        traffic_file_list = small_traffic_list
    elif args.traffic == 'debug':
        traffic_file_list = debug_traffic_list
    elif args.traffic == 'hangzhou':
        traffic_file_list = hangzhou_traffic_list
    elif args.traffic == 'jinan':
        traffic_file_list = jinan_traffic_list
    elif args.traffic == 'atlanta':
        traffic_file_list = atlanta_traffic_list
    elif args.traffic == 'three':
        traffic_file_list = three_traffic_list
    elif args.traffic == 'five':
        traffic_file_list = five_traffic_list
    else:
        raise(ValueError)
    return traffic_file_list

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def config_all(args):
    dic_traffic_env_conf_extra = {
        # file
        "ROADNET_FILE": args.roadnet,
        "FLOW_FILE": args.flow_file,

        # gui
        "IF_GUI": args.sumo_gui,
        "SAVEREPLAY": args.replay,

        "EPISODE_LEN": args.episode_len,

        "DONE_ENABLE": args.done,

        # different env (traffic or point)


        # normalization
        "REWARD_NORM": False,
        "INPUT_NORM": False,

        "NUM_ROW": 1,
        "NUM_COL": 1,

        # state & reward
        # "LIST_STATE_FEATURE": [ "cur_phase", "lane_num_vehicle"],
        "DIC_REWARD_INFO": {"sum_num_vehicle_been_stopped_thres1": -0.25},

        "LANE_NUM": {
            "LEFT": 1,
            "RIGHT": 0,
            "STRAIGHT": 1
        },

        "PHASE": [
            'WT_ET',
            'NT_ST',
            'WL_EL',
            'NL_SL',
            # 'WT_WL',
            # 'ET_EL',
            # 'NT_NL',
            # 'ST_SL',
        ],

        "LOG_DEBUG": args.debug,
        "NUM_GENERATOR": args.num_generator,
        'MODEL_NAME': args.algorithm,
    }

    # policy & agent config
    dic_agent_conf_extra = {
        "UPDATE_Q_BAR_FREQ": 5,
        # network

        "N_LAYER": 2,
        'NORM': 'None',
        'EPOCH': args.epoch,

        'PERIOD': 5,
        'ACTIVATION_FN': 'leaky_relu',
        'GRADIENT_CLIP': True,
        'CLIP_SIZE': args.clip_size,
        'PRE_TRAIN_MODEL_NAME': args.pre_train_model_name,

        'OPTIMIZER': 'sgd',

        #
        "ALPHA": args.alpha,
        "MIN_ALPHA": args.min_alpha,
        "ALPHA_DECAY_STEP": args.learning_rate_decay_step,
        'SEED': args.seed,

        "EPSILON": args.epsilon,
        "MIN_EPSILON": args.min_epsilon,

        #
        'SAMPLE_SIZE': args.sample_size,
        'UPDATE_START': args.update_start,  # 500,
        'UPDATE_PERIOD': args.update_period,  # 300,
        "TEST_PERIOD": args.test_period,
    }

    # path config
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", args.memo),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo),
        "PATH_TO_DATA": os.path.join("data", "tmp"),
        "PATH_TO_ERROR": os.path.join("errors", args.memo),
        "PATH_TO_GRADIENT": os.path.join("records", args.memo),
    }

    # experiment config
    dic_exp_conf_extra = {
        "EPISODE_LEN": args.episode_len,
        "TEST_EPISODE_LEN": args.test_episode_len,
        "MODEL_NAME": args.algorithm,  # Todo

        "NUM_ROUNDS": args.run_round,
        "NUM_GENERATORS": 3,

        "NUM_EPISODE": 1,

        "MODEL_POOL": False,
        "NUM_BEST_MODEL": 1,

        "PRETRAIN": False,
        "PRETRAIN_NUM_ROUNDS": 20,
        "PRETRAIN_NUM_GENERATORS": 15,

        "AGGREGATE": False,
        "DEBUG": False,
        "EARLY_STOP": False,
    }

    model_name = args.algorithm
    deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
    deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(model_name.upper())),
                                  dic_agent_conf_extra)
    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)
    return deploy_dic_exp_conf, deploy_dic_agent_conf, deploy_dic_traffic_env_conf, deploy_dic_path

def parse_roadnet(roadnetFile):
    roadnet = json.load(open(roadnetFile))
    lane_phase_info_dict ={}

    # many intersections exist in the roadnet and virtual intersection is controlled by signal
    for intersection in roadnet["intersections"]:
        if intersection['virtual']:
            continue
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                    "same_start_lane": [],
                                                     "end_lane": [],
                                                     "phase": [],
                                                     "phase_startLane_mapping": {},
                                                     "phase_sameStartLane_mapping": {},
                                                     "phase_roadLink_mapping": {}}
        road_links = intersection["roadLinks"]

        start_lane = []
        same_start_lane = []
        end_lane = []
        roadLink_lane_pair = {ri: [] for ri in
                              range(len(road_links))}  # roadLink includes some lane_pair: (start_lane, end_lane)
        roadLink_same_start_lane = {ri: [] for ri in
                              range(len(road_links))}  # roadLink includes some lane_pair: (start_lane, end_lane)

        for ri in range(len(road_links)):
            road_link = road_links[ri]
            tmp_same_start_lane = []
            for lane_link in road_link["laneLinks"]:
                sl = road_link['startRoad'] + "_" + str(lane_link["startLaneIndex"])
                el = road_link['endRoad'] + "_" + str(lane_link["endLaneIndex"])
                start_lane.append(sl)
                tmp_same_start_lane.append(sl)
                end_lane.append(el)
                roadLink_lane_pair[ri].append((sl, el))
            tmp_same_start_lane = tuple(set(tmp_same_start_lane))
            roadLink_same_start_lane[ri].append(tmp_same_start_lane)
            same_start_lane.append(tmp_same_start_lane)


        lane_phase_info_dict[intersection['id']]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]["end_lane"] = sorted(list(set(end_lane)))
        lane_phase_info_dict[intersection['id']]["same_start_lane"] = sorted(list(set(same_start_lane)))

        for phase_i in range(1, len(intersection["trafficLight"]["lightphases"])):
            p = intersection["trafficLight"]["lightphases"][phase_i]
            lane_pair = []
            start_lane = []
            same_start_lane = []
            for ri in p["availableRoadLinks"]:
                lane_pair.extend(roadLink_lane_pair[ri])
                for i in range(len(roadLink_lane_pair[ri])):
                    if roadLink_lane_pair[ri][i][0] not in start_lane:
                        start_lane.append(roadLink_lane_pair[ri][i][0])

                if roadLink_same_start_lane[ri][0] not in same_start_lane:
                    same_start_lane.append(roadLink_same_start_lane[ri][0])
            lane_phase_info_dict[intersection['id']]["phase"].append(phase_i)
            lane_phase_info_dict[intersection['id']]["phase_startLane_mapping"][phase_i] = start_lane
            lane_phase_info_dict[intersection['id']]["phase_sameStartLane_mapping"][phase_i] = same_start_lane
            lane_phase_info_dict[intersection['id']]["phase_roadLink_mapping"][phase_i] = lane_pair

    return lane_phase_info_dict