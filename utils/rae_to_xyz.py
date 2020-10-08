import numpy as np
import os
from tqdm import tqdm
import shutil

DATASET_DIR = '/home/juan/PycharmProjects/Trajectory-Transformer/datasets'

def rae_to_earth(position):
    """
    先将弧度转化为角度
    把RAE转换成以雷达站为坐标原点,北为Y轴正向,东为X轴正向的XYZ坐标
    """
    z = position[:, 2] * np.sin(position[:, 1])
    xy = position[:, 2] * np.cos(position[:, 1])
    y = xy * np.sin(position[:, 0])
    x = xy * np.cos(position[:, 0])
    for i in range(position.shape[0]):
        if position[i, 0] < 0:
            x[i] *= - 1
            y[i] *= -1
    return x[:], y[:], z[:]


def distance(x1, x2) -> float:
    """
    计算两个点之间的欧式距离
    """
    return np.linalg.norm(x1 - x2)


def cep(launch, stop, diameter: float) -> float:
    return distance(launch, stop) / diameter


def read_aer(rae_dir: str, output_dir: str = ''):
    """
    输入rae数据所在文件夹，将其转化为xyz坐标
    """
    assert rae_dir is not None
    if not output_dir:
        target_dir = os.path.join(DATASET_DIR, 'radar_10_2')
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.mkdir(os.path.join(DATASET_DIR, 'radar_10_2'))
        output_dir = target_dir
    file_lists = os.listdir(rae_dir)
    for file_name in tqdm(file_lists):
        if 'data' not in file_name:
            continue
        file_path = os.path.join(rae_dir, file_name)
        f = open(file_path, 'r')
        # 第一行是发射点坐标
        launch_point = f.readline()
        # 第二行是雷达站坐标
        radar_station = f.readline()
        curve = []
        # 读取所有点
        for line in f:
            tmp = list(map(float, line.split()[0:3]))
            curve.append(np.array(tmp))
        curve = np.stack(curve)
        x, y, z = rae_to_earth(curve)
        output_path = os.path.join(output_dir, file_name)
        output_file = open(output_path, 'a')
        output_file.write(launch_point)
        output_file.write(radar_station)
        # 写入所有的点
        for i in range(len(x)):
            tmp = [x[i], y[i], z[i]]
            output_file.write('\t'.join(map(str, tmp)) + '\n')
        f.close()
        output_file.close()


if __name__ == '__main__':
    read_aer('/home/juan/radar_data')
