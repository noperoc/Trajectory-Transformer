import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import sys
import numpy as np
from tqdm import tqdm, trange

def plot(data_dir):
    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    file_list = os.listdir(data_dir)
    file_list = sorted(file_list)
    curves = []
    for file in tqdm(file_list):
        cur_file = open(os.path.join(data_dir, file), 'r')
        cur_file.readline()
        cur_file.readline()
        curve = []
        for line in cur_file:
            curve.append(np.array(list(map(float, line.split('\t')[0:3]))))
        curve = np.stack(curve)
        curves.append(curve)
    for i in trange(len(curves)):
        ax.plot(curves[i][:, 0], curves[i][:, 1], curves[i][:, 2])

    plt.pause(2147483646)
    plt.show()


if __name__ == '__main__':
    plot('/home/juan/PycharmProjects/Trajectory-Transformer/datasets/radar_10_2')
