import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path


fontsize = 13
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = fontsize
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.titlesize'] = fontsize
rcParams['axes.labelsize'] = fontsize
# rcParams['text.usetex'] = True
rcParams["savefig.dpi"] = 150

fig_all, ax_all = plt.subplots(3)


def generate_vector_field(vector_field, title, ax_count):
    vector_field = vector_field.reset_index()
    x_min = vector_field['x'].min()
    x_max = vector_field['x'].max()
    vector_field['x'] = (vector_field['x'] - x_min) / (x_max - x_min)
    y_min = vector_field['y'].min()
    y_max = vector_field['y'].max()
    vector_field['y'] = (vector_field['y'] - y_min) / (y_max - y_min)
    vel_x_max = vector_field['velocity_x'].max()
    vel_y_max = vector_field['velocity_y'].max()
    vector_field['velocity_x'] = 0.01 * vector_field['velocity_x'] / \
        (vel_x_max ** 2 + vel_y_max ** 2) ** 0.5
    vector_field['velocity_y'] = 0.01 * vector_field['velocity_y'] / \
        (vel_x_max ** 2 + vel_y_max ** 2) ** 0.5
    vector_field['color'] = vector_field['done'].apply(
        lambda done: 'red' if done else 'yellow')
    vector_field_done = vector_field[vector_field['done'] == True]
    vector_field = vector_field[vector_field['done'] == False]
    vector_field = vector_field.iloc[::20, :]
    vector_field = pd.concat([vector_field, vector_field_done], ignore_index=True)
    ax_all[ax_count].quiver(
        vector_field['x'],
        1 - vector_field['y'],
        vector_field['velocity_x'],
        -1 * vector_field['velocity_y'],
        color=vector_field['color'],
        scale=1)
    ax_all[ax_count].xaxis.set_ticks([])
    ax_all[ax_count].yaxis.set_ticks([])
    ax_all[ax_count].axis([-0.1, 1.1, -0.1, 1.1])
    # max_ep_count = vector_field['ep_count'].max()
    # min_ep_count = vector_field['ep_count'].min() - 1
    # timesteps_per_episode = 7200
    ax_all[ax_count].set_title(title)


ax_count = 0
rcParams["savefig.dpi"] = 300
img_path = Path('paper_plots/vector_field/output')
img_path.mkdir(parents=True, exist_ok=True)
title = 'Early train'
train_vector_field = pd.read_csv(
    'runs/env_info/1/1.csv')
train_vector_field['eval_mode'] = train_vector_field['step_rollout'].diff()
train_vector_field['end_eval'] = train_vector_field['eval_mode'].diff() == 1
train_vector_field['eval_mode'] = train_vector_field['eval_mode'] != 1
# train_vector_field = train_vector_field[~train_vector_field['eval_mode']]
# train_vector_field['ep_end'] = (train_vector_field['step_rollout'] % 2048) == 0
# train_vector_field.to_csv('train_vector_field.csv')
# steps_end_eval = train_vector_field[train_vector_field['ep_end']].index.to_list()
# last_index = 0
# for eval_index in steps_end_eval:
#     print('Start index: ' + str(last_index))
#     print('End index: ' + str(eval_index))
#     df_slice = train_vector_field.iloc[last_index:eval_index]
#     df_end_episodes = df_slice[df_slice['done']]
#     eps_ends = df_end_episodes.groupby(['command'])['command'].count()
#     print(eps_ends)
#     last_index = eval_index

train_vector_field = train_vector_field[train_vector_field['eval_mode']]
eval_steps_lists = train_vector_field['step_rollout'].unique().tolist()
for eval_steps in eval_steps_lists:
    print('Evaluation Steps: ' + str(eval_steps))
    df_slice = train_vector_field[train_vector_field['step_rollout'] == eval_steps]
    df_end_episodes = df_slice[df_slice['done']]
    eps_ends = df_end_episodes.groupby(['command'])['command'].count()
    print(eps_ends)