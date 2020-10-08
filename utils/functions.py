import os
import random

import scipy.spatial
import torch
from torch import Tensor
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np

import baselineUtils
from individual_TF_3D import IndividualTF
from transformer_3d.functional import subsequent_mask
from baselineUtils import IndividualTf3DDataset


def load_model(model_path: str) -> IndividualTF:
    """
    加载一个模型
    """
    model = IndividualTF(enc_inp_size=3, dec_inp_size=4, dec_out_size=4)
    model.load_state_dict(torch.load(model_path))
    return model


def predict(model: IndividualTF,
            dataset: IndividualTf3DDataset,
            dataloader: DataLoader):
    """
    :param model 加载的模型
    :param dataset  数据集
    :param dataloader 对应的 DataLoader
    """

    # 必须要将模型设为评估模式
    model.eval()

    mean = torch.cat((dataset[:]['src'][:, 1:, 3:6], dataset[:]['trg'][:, :, 3:6]), 1).mean((0, 1))
    std = torch.cat((dataset[:]['src'][:, 1:, 3:6], dataset[:]['trg'][:, :, 3:6]), 1).std((0, 1))

    preds = dataset[:]['trg'][0].shape[0]

    gt = []
    pr = []
    input_ = []

    for i, batch in enumerate(dataloader):

        gt.append(batch['trg'][:, :, 0:3])
        input_.append(batch['src'])

        inp = (batch['src'][:, 1:, 3:6] - mean) / std
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1]))
        decoder_input = torch.Tensor([0, 0, 0, 1]).unsqueeze(
            0).unsqueeze(1).repeat(inp.shape[0], 1, 1)

        for i in range(preds):
            trg_att = subsequent_mask(decoder_input.shape[1]).repeat(
                decoder_input.shape[0], 1, 1)
            out = model(inp, decoder_input, src_att, trg_att)
            decoder_input = torch.cat(
                (decoder_input, out[:, -1:, :]), 1)

        preds_tr_b = (decoder_input[:, 1:, 0:3] * std + mean).detach().numpy().cumsum(
            1) + batch['src'][:, -1:, 0:3].numpy()

        pr.append(preds_tr_b)

    gt = np.concatenate(gt, 0)
    pr = np.concatenate(pr, 0)
    input_ = np.concatenate(input_, 0)
    mad, fad, errs = baselineUtils.distance_metrics(gt, pr)

    return mad, fad


def radar_file_filter(data_path: str, angle='', size=0, shuffle=True):
    """
    对雷达文件夹下的数据进行过滤。

    :param data_path 雷达数据所在文件夹（完整路径）
    :param angle 射角（如：15.00 ），未指定则返回所有文件名，否则返回指定角度的文件名。
    :param size 要加载的条数。未指定则返回指定射角的所有文件名。
    :param shuffle 随机加载
    """
    assert size >= 0

    file_list = os.listdir(data_path)
    if not angle:
        return file_list
    specific_angle_list = list(filter(lambda file_name: angle in file_name, file_list))
    if shuffle:
        random.shuffle(specific_angle_list)
    if size:
        specific_angle_list = specific_angle_list[:size]
    assert specific_angle_list != []
    return specific_angle_list


def load_data_of_specific_angle(data_path: str,
                                gt: int,
                                horizon: int,
                                angle: str,
                                delim='\t',
                                size=0):
    """
    加载指定角度的雷达数据。可根据需要，使用该函数加载不同角度、不同观测点、预测点、不同尺寸的数据。

    :param data_path 雷达数据所在的文件夹（绝对路径）
    :param gt 预测点数
    :param horizon 观测点数
    :param angle 要加载的角度
    :param delim 记录中数据分隔符
    :param size 要加载数据集的大小，默认值为0，表示加载所有的数据
    """
    file_list = radar_file_filter(data_path, angle, size)
    data_dt = []

    # 取样点数
    sample_number = gt + horizon

    # 雷达数据集
    launch_points = []
    curves = []

    print("\033[0;33mStart loading all data \033[0m")

    files_with_index = [(i, j) for i, j in enumerate(file_list)]

    for i_dt, dt in tqdm(files_with_index):

        with open(os.path.join(data_path, dt), 'r') as f:
            launch_point = map(float, f.readline().strip())
            radar_position = map(float, f.readline().strip())
            curve = []  # 记录整条轨迹
            for i, line in enumerate(f):
                strs = line.strip().split(delim)
                tmp = np.array(list(map(float, strs))).astype(np.float)
                curve.append(tmp)
            threshold = len(curve) * 2 // 3
            curve = curve[0:threshold]
            curve = np.stack(curve)
            launch_points.append(curve[0])
            curves.append(curve)
            # curves中包含的是完整轨迹
            # 抽取数据 2 / 3，sample_number个点
            margin = curve.shape[0] // sample_number
            sample = np.stack(curve[::margin]).astype(float)[0:(sample_number)]
            assert sample.shape[0] == sample_number

            d_data = np.stack(sample[:, 0:3])  # 一个轨迹采样
            # 轨迹随机平移
            # shift = [np.random.rand() * 50, np.random.rand() * 50]
            # d_data[:, :2] += shift

            data_dt.append(d_data)

    data_dt_np = np.stack(data_dt)
    # 对抽取到的轨迹做翻转
    data_dt_np = data_dt_np[:, ::-1, :]
    inp_speed = np.concatenate(
        (np.zeros((data_dt_np.shape[0], 1, 3)),
         data_dt_np[:, 1:, :] - data_dt_np[:, :-1, :]), 1)

    inp_norm = np.concatenate((data_dt_np, inp_speed), 2)

    # 逆序轨迹，先取雷达观测到的部分，再取 GT
    inp = inp_norm[:, 0:horizon, :]
    out = inp_norm[:, horizon:, :]

    return inp, out


def distance_metrics(gt, preds):
    """
    计算欧氏距离，三维和二维通用
    """
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(
                gt[i, j, 0:2], preds[i, j, 0:2])
    return errors.mean(), errors[:, -1].mean(), errors


def gen_test_data(data_path: str,
                  gt: int,
                  horizon: int,
                  angle: str,
                  delim='\t',
                  size=0):
    """
    测试数据生成
    """
    data = {}

    inp, out = load_data_of_specific_angle(data_path, gt, horizon, angle, delim, size)

    data['src'] = inp
    data['trg'] = out

    mean_train = data['src'].mean((0, 1))
    std_train = data['src'].std((0, 1))

    dataset = IndividualTf3DDataset(data, "test", mean=mean_train, std=std_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                             num_workers=0)
    return dataset, dataloader


def evaluate(model_path: str,
             data_path: str,
             gt: int,
             horizon: int,
             delim='\t',
             size=0):
    angle_list = ['15.00', '15.50', '16.00', '16.50', '17.00', '17.50', '18.00', '18.50', '19.00', '19.50', '20.00',
                  '20.50', '21.00', '21.50', '22.00', '22.50', '23.00', '23.50', '24.00', '24.50', '25.00', '28.00',
                  '31.00', '34.00', '37.00', '40.00']
    res = []
    for angle in angle_list:
        dataset, dataloader = gen_test_data(data_path, gt, horizon, angle, delim, size)
        model = load_model(model_path)
        mad, fad = predict(model, dataset, dataloader)
        res.append((mad, fad))
    print('\n'.join(map(str, res)))


if __name__ == '__main__':
    DATA_DIR = '/home/juan/PycharmProjects/Trajectory-Transformer/datasets/radar_10_2'
    MODEL_PATH = '/home/juan/PycharmProjects/Trajectory-Transformer/models/Individual/radar_obs-10_pred-2/00399.pth'
    # gen_test_data(DIR, gt=2, horizon=10, angle='15.50')
    print(evaluate(model_path=MODEL_PATH, data_path=DATA_DIR, gt=2, horizon=10))
