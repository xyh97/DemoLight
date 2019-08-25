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
        res_path = os.path.join(path, traffic, "test_results.csv")
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


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--memo', type=str, default='hangzhou')

    args = parse.parse_args()
    memo = args.memo

    new_test_summary(memo)