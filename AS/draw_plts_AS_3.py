import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot():
    sns.set_theme(style="darkgrid")
    
    # data files
    common_dir = './CSV/AS_3'
    file_dir = {'traj_num=500': [1, 2, 3], 'traj_num=1000': [4, 5, 6], 'traj_num=2000': [7, 8, 9],
                'traj_num=3000': [10, 11, 12], 'traj_num=4000': [13, 14, 15]}


    data_frame = pd.DataFrame()
    for alg, dir_name_list in file_dir.items():
        for dir_name in dir_name_list:
            csv_file_name = str(dir_name) + '.csv'
            csv_file_dir = os.path.join(common_dir, csv_file_name)
            print("Loading from: ", alg, csv_file_dir)

            temp_df = pd.read_csv(csv_file_dir)
            temp_step = np.array(temp_df['Step'])
            temp_value = np.array(temp_df['Value'])
            print("Average rwd across the episodes: ", np.mean(temp_value))
            temp_len = len(temp_step)

            mov_avg_agent = MovAvg(2)
            for i in range(temp_len):
                if temp_step[i] > 600:
                    break
                if temp_step[i] % 25 == 0:

                    data_frame = data_frame.append({'algorithm': alg, 'Training Episode': temp_step[i], 'Exploitability': mov_avg_agent.update(temp_value[i])}, ignore_index=True)

    print(data_frame)
    sns.set(font_scale=1.5)
    pal = sns.color_palette("tab10")
    g = sns.relplot(x="Training Episode", y="Exploitability", hue='algorithm', kind="line", data=data_frame, legend='brief', palette=pal)
    leg = g._legend
    leg.set_bbox_to_anchor([0.6, 0.75])  # coordinates of lower left of bounding box [0.75, 0.4], [0.75, 0.6], [0.55, 0.65]
    # g = sns.relplot(x="training step", y="mean reward", hue='algorithm', kind="line", data=data_frame)
    g.fig.set_size_inches(15, 6)

    ax = g.ax

    y_ticks = [0 + 500 * i for i in range(7)]
    ax.set_yticks(y_ticks)
    ax.set_ylim((0, 3000))

    plt.savefig(common_dir + 'Exp.png')


class MovAvg(object):

    def __init__(self, window_size=50): # 20, 20, 50
        self.window_size = window_size
        self.data_queue = []

    def set_window_size(self, num):
        self.window_size = num

    def clear_queue(self):
        self.data_queue = []

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            del self.data_queue[0]
        self.data_queue.append(data)
        return sum(self.data_queue) / len(self.data_queue)


if __name__ == '__main__':
    plot()


