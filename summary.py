import pickle as pkl
import os
import pandas as pd
import numpy as np
import json
import copy
from math import isnan
import matplotlib as mlp
from matplotlib.offsetbox import AnchoredText
from script import *
import math
import shutil

mlp.use("agg")
import matplotlib.pyplot as plt

font = {'size': 24}
mlp.rc('font', **font)

NAN_LABEL = -1


def get_metrics(duration_list, min_duration, min_duration_id,
                traffic_name, total_summary, mode_name, save_path, num_rounds, min_duration2=None, min_duration_log=None):
    validation_duration_length = 10
    minimum_round = 50 if num_rounds > 50 else 0
    duration_list = np.array(duration_list)

    # min_duration, min_duration_id = np.min(duration_list), np.argmin(duration_list)
    # min_queue_length, min_queue_length_id = np.min(queue_length_list), np.argmin(queue_length_list)

    nan_count = len(np.where(duration_list == NAN_LABEL)[0])
    validation_duration = duration_list[-validation_duration_length:]
    final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)
    final_duration_std = np.round(np.std(validation_duration[validation_duration > 0]), decimals=2)

    if nan_count == 0:
        convergence = {1.2: len(duration_list) - 1, 1.1: len(duration_list) - 1}
        for j in range(minimum_round, len(duration_list)):
            for level in [1.2, 1.1]:
                if max(duration_list[j:]) <= level * final_duration:
                    if convergence[level] > j:
                        convergence[level] = j
        conv_12 = convergence[1.2]
        conv_11 = convergence[1.1]
    else:
        conv_12, conv_11 = 0, 0

    # simple plot for each training instance
    f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
    ax.plot(duration_list, linewidth=2, color='k')
    ax.plot([0, len(duration_list)], [final_duration, final_duration], linewidth=2, color="g")

    ax.plot([conv_12, conv_12], [duration_list[conv_12], duration_list[conv_12] * 3], linewidth=2, color="b")
    ax.plot([conv_11, conv_11], [duration_list[conv_11], duration_list[conv_11] * 3], linewidth=2, color="b")
    ax.plot([0, len(duration_list)], [min_duration, min_duration], linewidth=2, color="r")
    ax.plot([min_duration_id, min_duration_id], [min_duration, min_duration * 3], linewidth=2, color="r")

    init_dur = str(int(duration_list[0])) if not math.isnan(duration_list[0]) else 'NaN'
    min_dur = str(int(min_duration)) if not math.isnan(min_duration) else 'NaN'
    print(traffic_name, final_duration)
    final_dur = str(int(final_duration)) if not math.isnan(final_duration)  else 'NaN'

    if min_duration == 'NaN':
        min_duration = str(int(min_duration))

    anchored_text = AnchoredText("Initial_duration: %s\nMin_duration: %s\nFinal_duration: %s"%(init_dur, min_dur, final_dur), loc=1)
    ax.add_artist(anchored_text)

    ax.set_title(traffic_name + "-" + str(final_duration))
    plt.savefig(save_path + "/" + traffic_name + "-" + mode_name + ".png")
    figure_2 = os.path.join(os.path.dirname(save_path), 'total_figures')
    if not os.path.exists(figure_2):
        os.makedirs(figure_2)

    traffic_file = traffic_name
    if ".xml" in traffic_file:
        traffic_name, traffic_time = traffic_file.split(".xml")
    elif ".json" in traffic_file:
        traffic_name, traffic_time = traffic_file.split(".json")
    plt.savefig(figure_2 + "/" + traffic_name + ".png")
    plt.close()

    total_summary["traffic_file"].append(traffic_name)
    total_summary["traffic"].append(traffic_name.split(".xml")[0])
    total_summary["min_duration"].append(min_duration)
    total_summary["min_duration_round"].append(min_duration_id)
    total_summary["final_duration"].append(final_duration)
    total_summary["final_duration_std"].append(final_duration_std)
    total_summary["convergence_1.2"].append(conv_12)
    total_summary["convergence_1.1"].append(conv_11)
    total_summary["nan_count"].append(nan_count)
    total_summary["min_duration2"].append(min_duration2)
    if min_duration_log:
        for i in range(len(min_duration_log)):
            total_summary['md_%d'%((i + 1) * 10)].append(min_duration_log[i][0])
            total_summary['md_ind_%d' %((i + 1) * 10)].append(min_duration_log[i][1])


    return total_summary


def summary_plot(traffic_performance, figure_dir, mode_name, num_rounds):
    minimum_round = 50 if num_rounds > 50 else 0
    validation_duration_length = 10
    anomaly_threshold = 1.3

    for traffic_name in traffic_performance:
        f, ax = plt.subplots(2, 1, figsize=(12, 9), dpi=100)
        performance_tmp = []
        check_list = []
        for ti in range(len(traffic_performance[traffic_name])):
            ax[0].plot(traffic_performance[traffic_name][ti][0], linewidth=2)
            validation_duration = traffic_performance[traffic_name][ti][0][-validation_duration_length:]
            final_duration = np.round(np.mean(validation_duration), decimals=2)
            if len(np.where(traffic_performance[traffic_name][ti][0] == NAN_LABEL)[0]) == 0:
                # and len(traffic_performance[traffic_name][ti][0]) == num_rounds:
                tmp = traffic_performance[traffic_name][ti][0]
                if len(tmp) < num_rounds:
                    tmp.extend([float("nan")] * (num_rounds - len(traffic_performance[traffic_name][ti][0])))
                performance_tmp.append(tmp)
                check_list.append(final_duration)
            else:
                print("the length of traffic {} is shorter than {}".format(traffic_name, num_rounds))
        check_list = np.array(check_list)
        for ci in np.where(check_list > anomaly_threshold * np.mean(check_list))[0][::-1]:
            del performance_tmp[ci]
            print("anomaly traffic_name:{} id:{} err:{}".format(traffic_name, ci, check_list[ci] - np.mean(check_list)))
        if len(performance_tmp) == 0:
            print("The result of {} is not enough for analysis.".format(traffic_name))
            continue
        try:
            performance_summary = np.array(performance_tmp)
            print(traffic_name, performance_summary.shape)
            ax[1].errorbar(x=range(len(traffic_performance[traffic_name][0][0])),
                           y=np.mean(performance_summary, axis=0),
                           yerr=np.std(performance_summary, axis=0))

            psm = np.mean(performance_summary, axis=0)
            validation_duration = psm[-validation_duration_length:]
            final_duration = np.round(np.mean(validation_duration), decimals=2)

            convergence = {1.2: len(psm) - 1, 1.1: len(psm) - 1}
            for j in range(minimum_round, len(psm)):
                for level in [1.2, 1.1]:
                    if max(psm[j:]) <= level * final_duration:
                        if convergence[level] > j:
                            convergence[level] = j
            ax[1].plot([0, len(psm)], [final_duration, final_duration], linewidth=2, color="g")
            ax[1].text(len(psm), final_duration * 2, "final-" + str(final_duration))
            ax[1].plot([convergence[1.2], convergence[1.2]], [psm[convergence[1.2]], psm[convergence[1.2]] * 3],
                       linewidth=2, color="b")
            ax[1].text(convergence[1.2], psm[convergence[1.2]] * 2, "conv 1.2-" + str(convergence[1.2]))
            ax[1].plot([convergence[1.1], convergence[1.1]], [psm[convergence[1.1]], psm[convergence[1.1]] * 3],
                       linewidth=2, color="b")
            ax[1].text(convergence[1.1], psm[convergence[1.1]] * 2, "conv 1.1-" + str(convergence[1.1]))
            ax[1].set_title(traffic_name)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
            plt.savefig(figure_dir + "/" + traffic_name + "-" + mode_name + ".png")
            plt.close()
        except:
            print("plot error")


def plot_segment_duration(round_summary, path, mode_name):
    save_path = os.path.join(path, "segments")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for key in round_summary.keys():
        if "duration" in key:
            f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
            ax.plot(round_summary[key], linewidth=2, color='k')
            ax.set_title(key)
            plt.savefig(save_path + "/" + key + "-" + mode_name + ".png")
            plt.close()


def padding_duration(performance_duration):
    for traffic_name in performance_duration.keys():
        max_duration_length = max([len(x[0]) for x in performance_duration[traffic_name]])
        for i, ti in enumerate(performance_duration[traffic_name]):
            performance_duration[traffic_name][i][0].extend((max_duration_length - len(ti[0]))*[ti[0][-1]])

    return performance_duration


def performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name):
    for traffic_name in performance_at_min_duration_round:
        f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
        for ti in range(len(performance_at_min_duration_round[traffic_name])):
            ax.plot(performance_at_min_duration_round[traffic_name][ti][0], linewidth=2)
        plt.savefig(figure_dir + "/" + "min_duration_round" + "-" + mode_name + ".png")
        plt.close()


def summary_detail_train(memo, total_summary):
    # each_round_train_duration

    performance_duration = {}
    performance_at_min_duration_round = {}
    records_dir = os.path.join("records", memo)
    for traffic_file in os.listdir(records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue
        print(traffic_file)

        min_queue_length = min_duration = float('inf')
        min_queue_length_id = min_duration_ind = 0

        # get episode_len to calculate the queue_length each second
        exp_conf = open(os.path.join(records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        episode_len = dic_exp_conf["EPISODE_LEN"]
        num_rounds = dic_exp_conf["NUM_ROUNDS"]
        num_seg = episode_len // 3600

        traffic_vol = get_total_traffic_volume(dic_exp_conf["TRAFFIC_FILE"][0])
        nan_thres = 120

        duration_each_round_list = []
        queue_length_each_round_list = []

        train_round_dir = os.path.join(records_dir, traffic_file, "train_round")
        round_files = os.listdir(train_round_dir)
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[6:]))
        round_summary = {"round": list(range(num_rounds))}
        for round in round_files:
            try:
                round_dir = os.path.join(train_round_dir, round)

                duration_gens = 0
                queue_length_gens = 0
                cnt_gen = 0

                list_duration_seg = [float('inf')] * num_seg
                list_queue_length_seg = [float('inf')] * num_seg
                list_queue_length_id_seg = [0] * num_seg
                list_duration_id_seg = [0] * num_seg
                for gen in os.listdir(round_dir):
                    if "generator" not in gen:
                        continue

                    # summary items (queue_length) from pickle
                    gen_dir = os.path.join(records_dir, traffic_file, "train_round", round, gen)
                    f = open(os.path.join(gen_dir, "inter_0.pkl"), "rb")
                    samples = pkl.load(f)

                    for sample in samples:
                        queue_length_gens += sum(sample['state']['lane_queue_length'])
                    sample_num = len(samples)
                    f.close()

                    # summary items (duration) from csv
                    df_vehicle_inter_0 = pd.read_csv(os.path.join(round_dir, gen, "vehicle_inter_0.csv"),
                                                     sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                     names=["vehicle_id", "enter_time", "leave_time"])

                    duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
                    ave_duration = np.mean([time for time in duration if not isnan(time)])
                    real_traffic_vol = 0
                    nan_num = 0
                    for time in duration:
                        if not isnan(time):
                            real_traffic_vol += 1
                        else:
                            nan_num += 1
                    # print(ave_duration)

                    cnt_gen += 1
                    duration_gens += ave_duration

                    for i, interval in enumerate(range(0, episode_len, 3600)):
                        did = np.bitwise_and(df_vehicle_inter_0["enter_time"].values < interval + 3600,
                                             df_vehicle_inter_0["enter_time"].values > interval)
                        # vehicle_in_seg = sum([int(x) for x in (df_vehicle_inter_0["enter_time"][did].values > 0)])
                        # vehicle_out_seg = sum([int(x) for x in (df_vehicle_inter_0["leave_time"][did].values > 0)])
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

                        # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)

                        if nan_num_seg < nan_thres:
                            # if min_duration[i] > ave_duration and ave_duration > 24:
                            list_duration_seg[i] = ave_duration_seg
                            list_duration_id_seg[i] = int(round[6:])


                list_duration_seg = np.array(list_duration_seg)/cnt_gen
                for j in range(num_seg):
                    key = "min_duration-" + str(j)
                    if key not in round_summary.keys():
                        round_summary[key] = [list_duration_seg[j]]
                    else:
                        round_summary[key].append(list_duration_seg[j])


                duration_each_round_list.append(duration_gens / cnt_gen)
                queue_length_each_round_list.append(queue_length_gens / cnt_gen / sample_num)

                # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)
                if min_queue_length > queue_length_gens / cnt_gen / sample_num:
                    min_queue_length = queue_length_gens / cnt_gen / sample_num
                    min_queue_length_id = int(round[6:])

                valid_flag = json.load(open(os.path.join(gen_dir, "valid_flag.json")))
                if valid_flag['0']: # temporary for one intersection
                    if min_duration > duration_gens / cnt_gen:
                        min_duration = duration_gens / cnt_gen
                        min_duration_ind = int(round[6:])
                #print(nan_num, nan_thres)

            except:
                # change anomaly label from nan to -1000 for the convenience of following computation
                duration_each_round_list.append(NAN_LABEL)
                queue_length_each_round_list.append(NAN_LABEL)

        result_dir = os.path.join("summary", memo, traffic_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(result_dir, "train_results.csv"))
        if num_seg > 1:
            round_result = pd.DataFrame(round_summary)
            round_result.to_csv(os.path.join(result_dir, "train_seg_results.csv"), index=False)
            # plot duration segment
            plot_segment_duration(round_summary, result_dir, mode_name="train")
            duration_each_segment_list = round_result.iloc[min_duration_ind][1:].values
            if ".xml" in traffic_file:
                traffic_name, traffic_time = traffic_file.split(".xml")
            elif ".json" in traffic_file:
                traffic_name, traffic_time = traffic_file.split(".json")
            if traffic_name not in performance_at_min_duration_round:
                performance_at_min_duration_round[traffic_name] = [(duration_each_segment_list, traffic_time)]
            else:
                performance_at_min_duration_round[traffic_name].append((duration_each_segment_list, traffic_time))



        # total_summary
        total_summary = get_metrics(duration_each_round_list, queue_length_each_round_list,
                                    min_duration, min_duration_ind, min_queue_length, min_queue_length_id,
                                    traffic_file, total_summary,
                                    mode_name="train", save_path=result_dir, num_rounds=num_rounds)

        if ".xml" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".xml")
        elif ".json" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".json")
        if traffic_name not in performance_duration:
            performance_duration[traffic_name] = [(duration_each_round_list, traffic_time)]
        else:
            performance_duration[traffic_name].append((duration_each_round_list, traffic_time))

    figure_dir = os.path.join("summary", memo, "figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    if dic_exp_conf["EARLY_STOP"]:
        performance_duration = padding_duration(performance_duration)
    summary_plot(performance_duration, figure_dir, mode_name="train", num_rounds=num_rounds)
    total_result = pd.DataFrame(total_summary)
    total_result.to_csv(os.path.join("summary", memo, "total_train_results.csv"))
    performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name="train")


def summary_detail_test(memo, total_summary, num_log):
    # each_round_train_duration

    performance_duration = {}
    performance_at_min_duration_round = {}
    records_dir = os.path.join("records", memo)

    for traffic_file in os.listdir(records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue

        #if "cross.2phases_rou01_equal_700.xml_12_11_08_16_00" != traffic_file:
        #    continue
        print(traffic_file)

        min_duration = min_duration2 = float('inf')
        min_duration_ind = 0

        # get episode_len to calculate the queue_length each second
        exp_conf = open(os.path.join(records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        episode_len = dic_exp_conf["EPISODE_LEN"]
        num_rounds = dic_exp_conf["NUM_ROUNDS"]
        num_seg = episode_len//3600

        traffic_vol = get_total_traffic_volume(dic_exp_conf["TRAFFIC_FILE"][0])
        nan_thres = 120

        duration_each_round_list = []
        duration_each_round_list2 = []
        num_of_vehicle_in = []
        num_of_vehicle_out = []

        train_round_dir = os.path.join(records_dir, traffic_file, "test_round", 'task_0_'+dic_exp_conf["TRAFFIC_FILE"])
        try:
            round_files = os.listdir(train_round_dir)
        except:
            print("no test round in {}".format(traffic_file))
            continue
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[12:]))
        round_summary = {"round": list(range(num_rounds))}

        min_duration_log = []

        for round_id in range(len(round_files)):
            round = round_files[round_id]
            #try:
            if 1:
                round_dir = os.path.join(train_round_dir, round)

                list_duration_seg = [float('inf')] * num_seg
                list_queue_length_seg = [float('inf')] * num_seg
                list_queue_length_id_seg = [0] * num_seg
                list_duration_id_seg = [0] * num_seg

                # summary items (queue_length) from pickle

                # summary items (duration) from csv
                df_vehicle_inter_0 = pd.read_csv(os.path.join(round_dir, "vehicle_inter_0.csv"),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])

                vehicle_in = sum([int(x) for x in (df_vehicle_inter_0["enter_time"].values > 0)])
                vehicle_out = sum([int(x) for x in (df_vehicle_inter_0["leave_time"].values > 0)])
                total_vol = get_total_traffic_volume(traffic_file)
                duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
                ave_duration = np.mean([time for time in duration if not isnan(time)])
                # print(ave_duration)

                if "peak" in traffic_file:
                    did1 = df_vehicle_inter_0["enter_time"].values <= episode_len / 2
                    duration = df_vehicle_inter_0["leave_time"][did1].values - df_vehicle_inter_0["enter_time"][
                        did1].values
                    ave_duration = np.mean([time for time in duration if not isnan(time)])

                    did2 = df_vehicle_inter_0["enter_time"].values > episode_len / 2
                    duration2 = df_vehicle_inter_0["leave_time"][did2].values - df_vehicle_inter_0["enter_time"][
                        did2].values
                    ave_duration2 = np.mean([time for time in duration2 if not isnan(time)])
                    duration_each_round_list2.append(ave_duration2)

                    real_traffic_vol2 = 0
                    nan_num2 = 0
                    for time in duration2:
                        if not isnan(time):
                            real_traffic_vol2 += 1
                        else:
                            nan_num2 += 1

                    if nan_num2 < nan_thres:
                        if min_duration2 > ave_duration2 and ave_duration2 > 24:
                            min_duration2 = ave_duration2
                            min_duration_ind2 = int(round[6:])

                real_traffic_vol = 0
                nan_num = 0
                for time in duration:
                    if not isnan(time):
                        real_traffic_vol += 1
                    else:
                        nan_num += 1

                # ********* new calculation of duration **********
                flow_file = traffic_file[:traffic_file.find('.json')+5]

                df_vehicle_planed_enter = get_planed_entering(os.path.join(records_dir, traffic_file, flow_file), episode_len)
                ave_duration = cal_travel_time(df_vehicle_inter_0, df_vehicle_planed_enter, episode_len)

                #if vehicle_out < total_vol * 0.9:
                #    ave_duration = (ave_duration * vehicle_out + 800 * (total_vol - vehicle_out)) / total_vol

                duration_each_round_list.append(ave_duration)
                num_of_vehicle_in.append(vehicle_in)
                num_of_vehicle_out.append(vehicle_out)

                if min_duration > ave_duration:
                    print(">", traffic_file)
                    print(">>>", ave_duration, vehicle_out, total_vol)
                    min_duration = ave_duration
                    min_duration_ind = int(round[12:])

                if round_id % 10 == 9 and len(min_duration_log) <= num_log - 1:
                    min_duration_log.append((min_duration, min_duration_ind))

                if num_seg > 1:
                    for i, interval in enumerate(range(0, episode_len, 3600)):
                        did = np.bitwise_and(df_vehicle_inter_0["enter_time"].values < interval + 3600,
                                             df_vehicle_inter_0["enter_time"].values > interval)
                        #vehicle_in_seg = sum([int(x) for x in (df_vehicle_inter_0["enter_time"][did].values > 0)])
                        #vehicle_out_seg = sum([int(x) for x in (df_vehicle_inter_0["leave_time"][did].values > 0)])
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

                        # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)

                        if nan_num_seg < nan_thres:
                            # if min_duration[i] > ave_duration and ave_duration > 24:
                            list_duration_seg[i] = ave_duration_seg
                            list_duration_id_seg[i] = int(round[6:])

                        #round_summary = {}
                    for j in range(num_seg):
                        key = "min_duration-" + str(j)
                        if key not in round_summary.keys():
                            round_summary[key] = [list_duration_seg[j]]
                        else:
                            round_summary[key].append(list_duration_seg[j])
                    #round_result_dir = os.path.join("summary", memo, traffic_file)
                    #if not os.path.exists(round_result_dir):
                    #    os.makedirs(round_result_dir)

            #except:
            #    duration_each_round_list.append(NAN_LABEL)
            #    num_of_vehicle_in.append(NAN_LABEL)
            #    num_of_vehicle_out.append(NAN_LABEL)
            #    if "peak" in traffic_file:
            #        duration_each_round_list2.append(NAN_LABEL)

        # result_dir = os.path.join(records_dir, traffic_file)
        result_dir = os.path.join("summary", memo, traffic_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        _res = {
            "duration": duration_each_round_list,
            "vehicle_in": num_of_vehicle_in,
            "vehicle_out": num_of_vehicle_out
        }
        result = pd.DataFrame(_res)

        total_results_path = os.path.join("summary", memo, 'total_results')
        if not os.path.exists(total_results_path):
            os.makedirs(total_results_path)

        result.to_csv(os.path.join(result_dir, "test_results.csv"))
        ## csv
        if ".xml" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".xml")
        elif ".json" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".json")
        result.to_csv(os.path.join(total_results_path, traffic_name + ".csv"))
        if num_seg > 1:
            round_result = pd.DataFrame(round_summary)
            round_result.to_csv(os.path.join(result_dir, "test_seg_results.csv"), index=False)
            plot_segment_duration(round_summary, result_dir, mode_name="test")
            duration_each_segment_list = round_result.iloc[min_duration_ind][1:].values

            traffic_name, traffic_time = traffic_file.split(".xml")
            if traffic_name not in performance_at_min_duration_round:
                performance_at_min_duration_round[traffic_name] = [(duration_each_segment_list, traffic_time)]
            else:
                performance_at_min_duration_round[traffic_name].append((duration_each_segment_list, traffic_time))


        # print(os.path.join(result_dir, "test_results.csv"))

        # total_summary
        total_summary = get_metrics(duration_each_round_list,
                                    min_duration, min_duration_ind, traffic_file, total_summary,
                                    mode_name="test", save_path=result_dir, num_rounds=num_rounds,
                                    min_duration_log=min_duration_log,
                                    min_duration2=None if "peak" not in traffic_file else min_duration2
                                    )

        if ".xml" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".xml")
        elif ".json" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".json")
        if traffic_name not in performance_duration:
            performance_duration[traffic_name] = [(duration_each_round_list, traffic_time)]
        else:
            performance_duration[traffic_name].append((duration_each_round_list, traffic_time))

    print(total_summary)
    total_result = pd.DataFrame(total_summary)

    total_result.to_csv(os.path.join("summary", memo, "total_test_results.csv"))
    figure_dir = os.path.join("summary", memo, "figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    if dic_exp_conf["EARLY_STOP"]:
        performance_duration = padding_duration(performance_duration)

    summary_plot(performance_duration, figure_dir, mode_name="test", num_rounds=num_rounds)
    performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name="test")

def get_planed_entering(flowFile, episode_len):
    # todo--check with huichu about how each vehicle is inserted, according to the interval. 1s error may occur.
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
            #dic_traj["flow_{0}_{1}".format(flow_id, vec_id)] = {"planed_enter_time": ts}

    df = pd.DataFrame(dic_traj)
    #df.set_index('vehicle_id')
    return df
    #return pd.DataFrame(dic_traj).transpose()

def cal_travel_time(df_vehicle_actual_enter_leave, df_vehicle_planed_enter, episode_len):
    df_vehicle_planed_enter.set_index('vehicle_id', inplace=True)
    df_vehicle_actual_enter_leave.set_index('vehicle_id', inplace=True)
    df_res = pd.concat([df_vehicle_planed_enter, df_vehicle_actual_enter_leave], axis=1, sort=False)
    assert len(df_res) == len(df_vehicle_planed_enter)

    df_res["leave_time"].fillna(episode_len, inplace=True)
    df_res["travel_time"] = df_res["leave_time"] - df_res["planed_enter_time"]
    travel_time = df_res["travel_time"].mean()
    return travel_time

def new_test_summary(memo):
    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }

    path = os.path.join("records", memo)
    for traffic in os.listdir(path):
        traffic_name = traffic[:traffic.find(".json") + len(".json")]
        task_name = "task_0_" + traffic_name
        res_path = os.path.join(path, traffic, "test_round", task_name, "test_results.csv")
        res_summary_path = os.path.join("summary", memo, "total_results")
        fig_summary_path = os.path.join("summary", memo, "total_figures")
        figures_path = os.path.join("summary", memo, "figures")
        if not os.path.exists(res_summary_path):
            os.makedirs(res_summary_path)
        if not os.path.exists(fig_summary_path):
            os.makedirs(fig_summary_path)
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        shutil.copy(res_path, os.path.join(res_summary_path, traffic[:traffic.find(".json")] + ".csv"))
        df = pd.read_csv(res_path)
        duration = df["duration"]
        min_duration = duration.min()
        min_duration_ind = duration[duration == min_duration].index[0]
        total_summary = get_metrics(duration,
                                    min_duration, min_duration_ind, traffic_name, total_summary,
                                    mode_name="test", save_path=figures_path, num_rounds=duration.size,
                                    )
    total_result = pd.DataFrame(total_summary)

    total_result.to_csv(os.path.join("summary", memo, "total_test_results.csv"))



def summary_detail_baseline(memo):
    # each_round_train_duration
    total_summary = {
        "traffic": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": []
    }

    records_dir = os.path.join("records", memo)
    for traffic_file in os.listdir(records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue
        print(traffic_file)

        # if "650" not in traffic_file:
        #    continue

        # get episode_len to calculate the queue_length each second
        exp_conf = open(os.path.join(records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        episode_len = dic_exp_conf["EPISODE_LEN"]

        duration_each_round_list = []
        queue_length_each_round_list = []

        train_dir = os.path.join(records_dir, traffic_file)

        # summary items (queue_length) from pickle
        f = open(os.path.join(train_dir, "inter_0.pkl"), "rb")
        try:
            samples = pkl.load(f)
        except:
            continue
        for sample in samples:
            queue_length_each_round = sum(sample['state']['lane_queue_length'])
        f.close()

        # summary items (duration) from csv
        df_vehicle_inter_0 = pd.read_csv(os.path.join(train_dir, "vehicle_inter_0.csv"),
                                         sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                         names=["vehicle_id", "enter_time", "leave_time"])

        duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
        ave_duration = np.mean([time for time in duration if not isnan(time)])
        # print(ave_duration)

        duration_each_round_list.append(ave_duration)
        ql = queue_length_each_round / len(samples)
        queue_length_each_round_list.append(ql)

        # result_dir = os.path.join(records_dir, traffic_file)
        result_dir = os.path.join("summary", memo, traffic_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(result_dir, "test_results.csv"))
        # print(os.path.join(result_dir, "test_results.csv"))

        total_summary["traffic"].append(traffic_file)
        total_summary["min_queue_length"].append(ql)
        total_summary["min_queue_length_round"].append(0)
        total_summary["min_duration"].append(ave_duration)
        total_summary["min_duration_round"].append(0)

    total_result = pd.DataFrame(total_summary)
    total_result.to_csv(os.path.join("summary", memo, "total_baseline_test_results.csv"))


def main(memo=None):
    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }
    if not memo:
        memo = "pipeline_500"
    #summary_detail_train(memo, copy.deepcopy(total_summary))
    summary_detail_test(memo, copy.deepcopy(total_summary))
    # summary_detail_test_segments(memo, copy.deepcopy(total_summary))


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--memo', type=str, default='default')
    parse.add_argument('--max_round', type=int, default=5)

    args = parse.parse_args()

    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }
    for i in range(1, args.max_round+1):
        total_summary['md_%d' % (i * 10)] = []
        total_summary['md_ind_%d' % (i * 10)] = []
    memo = args.memo #"0215_large_traffic_4_phase_pre_train_someone/lr_0.001"
    #summary_detail_train(memo, copy.deepcopy(total_summary))
    #summary_detail_test(memo, total_summary, num_log=args.max_round)

    new_test_summary(memo)
    # summary_detail_baseline(memo)
    #summary_detail_test_segments(memo, copy.deepcopy(total_summary))