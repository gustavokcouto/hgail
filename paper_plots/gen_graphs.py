from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


fontsize = 13
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = fontsize
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.titlesize'] = fontsize
rcParams['axes.labelsize'] = fontsize
rcParams['text.usetex'] = True
rcParams["savefig.dpi"] = 150

all_results_dir = Path('paper_plots/results')
results_subdirs = ['bc-gail-gan', 'bc-gail', 'bc-gan', 'rgb']
results_labels = {'bc-gail-gan': 'hGAIL', 'bc-gail': 'GAIL w/ real BEV', 'bc-gan': 'BC', 'rgb': 'GAIL from cam'}  # 
results_types = ['eval_episodes', 'eval_meters', 'eval_steps', 'rollout_episodes', 'rollout_meters', 'rollout_steps']
columns_names = ['eval/completed_n_episodes', 'eval/route_completed_in_m', 'eval/step', 'rollout/completed_n_episodes', 'rollout/route_completed_in_m', 'rollout/step']
columns_labels = {'eval/completed_n_episodes': 'infractions', 'eval/route_completed_in_m': 'route completed in m', 'eval/step': 'steps completed', 'rollout/completed_n_episodes': 'infractions', 'rollout/route_completed_in_m': 'route completed in m', 'rollout/step': 'step'}
for result_type, column_name in zip(results_types, columns_names):
    for results_subdir in results_subdirs:
        results_len = 3
        if results_subdir == 'bc-gan' and not result_type in ['eval_episodes', 'eval_meters', 'eval_steps']:
            continue
        if results_subdir == 'rgb' and not result_type in ['rollout_episodes', 'rollout_meters', 'rollout_steps', 'eval_steps']:
            continue
        results_dfs = []
        for i_result in range(results_len):
            results_dir = all_results_dir / f'{results_subdir}-{i_result}'
            results_file = results_dir / f'{result_type}.csv'
            results_df = pd.read_csv(results_file)
            results_dfs.append(results_df)
        
        min_length = None
        for i_result in range(results_len):
            steps_list = results_dfs[i_result]['Step'].tolist()
            if min_length is None:
                min_length = len(steps_list)
            else:
                min_length = min(len(steps_list), min_length)
            print(min_length)
            
        results_lists = []
        for i_result in range(results_len):
            steps_list = results_dfs[i_result]['Step'] / 100000
            steps_list = steps_list
            steps_list = steps_list[:min_length]
            result_column = results_dfs[i_result][column_name].tolist()
            result_column = result_column[:min_length]
            results_lists.append(result_column)

        results_array = np.array(results_lists)
        results_mean = results_array.mean(axis=0)
        results_std = results_array.std(axis=0)
        plt.plot(steps_list, results_mean, label=results_labels[results_subdir])
        plt.fill_between(steps_list, results_mean + results_std, results_mean - results_std, alpha=0.5)

    plt.legend(loc='best', shadow=True, fontsize='medium')
    plt.xlabel(r'environment interactions ($ \times 10^5$)')
    plt.ylabel(columns_labels[column_name])
    plt.savefig(f'{result_type}.png')
    plt.clf()

#     results_0 = pd.read_csv(results_files[0])
#     results_1 = pd.read_csv(results_files[1])

#     min_size = 420
#     results_list = [results_0, results_1]
#     for results in results_list:
#         min_size = min(min_size, len(results))

#     results_arr_list = []
#     for results in results_list:
#         results_arr = results['Value'].to_numpy()
#         results_arr = results_arr[:min_size]
#         results_arr_list.append(results_arr)

#     results = np.array(results_arr_list)
#     # results = results.reshape(-1, 3)
#     results_mean = results.mean(axis=0)
#     results_std = results.std(axis=0)
#     results_id = np.arange(results.shape[1]) * 7200 /100000

#     plt.plot(results_id, results_mean, label=results_files[4])
#     plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

# results_mean.fill(173.6)
# plt.plot(results_id, results_mean, label='bc')
# results_std.fill(137.35516007780703)
# plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

# plt.xlabel(r'environment interactions ($ \times 10^5$)')
# plt.ylabel('Reward')
# plt.legend(loc='lower right', shadow=True, fontsize='medium')
# plt.savefig('paper_plots/plots/long_train_reward.png')
# plt.clf()

# for results_files in results_dataset:
#     results_0 = pd.read_csv(results_files[2])
#     results_1 = pd.read_csv(results_files[3])

#     min_size = 420
#     results_list = [results_0, results_1]
#     for results in results_list:
#         min_size = min(min_size, len(results))

#     results_arr_list = []
#     for results in results_list:
#         results_arr = results['Value'].to_numpy()
#         results_arr = results_arr[:min_size]
#         results_arr_list.append(results_arr)

#     results = np.array(results_arr_list)
#     # results = results.reshape(-1, 3)
#     results_mean = results.mean(axis=0)
#     results_std = results.std(axis=0)
#     results_id = np.arange(results.shape[1]) * 7200 /100000

#     plt.plot(results_id, results_mean, label=results_files[4])
#     plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

# plt.xlabel(r'environment interactions ($ \times 10^5$)')
# plt.ylabel('Reward')
# plt.savefig('paper_plots/plots/long_eval_reward.png')