from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io
from pandas import DataFrame
from tqdm import *
from graphviz import Digraph
from sklearn.model_selection import train_test_split

PROJECT_FOLDER = '/home/juan/PycharmProjects/Trajectory-Transformer'


# 一些工具方法

def create_dataset(dataset_folder, dataset_name, val_size, gt, horizon, delim="\t", train=True, eval=False,
                   verbose=False):
    """
    创建dataset，文件夹数据集格式如下：test | train | val ，其中包含了数据预处理的部分。训练数据是从 train 文件夹下加载，以此类推。

    :param dataset_folder 数据集所在文件夹
    :param dataset_name   数据集名称
    :param val_size       验证集大小
    :param gt             TODO 对应 train 脚本的 obs 参数，默认为8，每一条训练数据的长度
    :param horizon        TODO
    :param delim          分隔符
    :param train          是否是训练集
    :param eval           是否是测试集
    """
    datasets_list = None
    full_dt_folder = None
    if train:
        # 生成训练集数据
        # datasets/eth/train 文件夹下的文件列表
        datasets_list = os.listdir(os.path.join(
            dataset_folder, dataset_name, "train"))
        # datasets/eth/train
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "train")
    if not train and not eval:
        # 生成验证集数据
        datasets_list = os.listdir(os.path.join(
            dataset_folder, dataset_name, "val"))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "val")
    if not train and eval:
        # 生成测试集数据
        datasets_list = os.listdir(os.path.join(
            dataset_folder, dataset_name, "test"))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "test")

    datasets_list = datasets_list
    data = {}
    data_src = []
    data_trg = []
    data_seq_start = []
    data_frames = []
    data_dt = []
    data_peds = []

    val_src = []
    val_trg = []
    val_seq_start = []
    val_frames = []
    val_dt = []
    val_peds = []

    if verbose:
        print("\033[0;31mstart loading dataset...\033[0m")
        print("validation set size -> %i" % val_size)

    for i_dt, dt in enumerate(datasets_list):
        if verbose:
            print("%03i / %03i - loading %s" %
                  (i_dt + 1, len(datasets_list), dt))
        # 读取相应数据集下的文件，并将其转换成DataFrame
        # 并设置相应的分隔符，并且设置相应的列名、列编号，并设置空值为 `?`
        raw_data = pd.read_csv(os.path.join(full_dt_folder, dt), delimiter=delim,
                               names=["frame", "ped", "x", "y"], usecols=[0, 1, 2, 3], na_values="?")

        raw_data.sort_values(by=['frame', 'ped'], inplace=True)

        # inp 的维度是 (8988, 8, 4)，其中 8 代表着每个行人对应的 8 个点（？？？），TODO 4 代表什么？一个可能的想法是原始数据配上归一化之后的数据。
        # info 中包含了一些统计信息，比如 平均值、方差、行人的 ID 以及帧号
        inp, out, info = get_strided_data_clust(raw_data, gt, horizon, 1)

        dt_frames = info['frames']  # (8988, 20)，生成了每一个序列对应的时间戳
        dt_seq_start = info['seq_start']  # (8988, 1, 2) 记录了每一个序列的起始位置
        # 重复 shape[0]，也就是输入数据的组数，标识的是所对应的数据集编号
        dt_dataset = np.array([i_dt]).repeat(inp.shape[0])
        dt_peds = info['peds']  # 记录每组数所对应的行人编号

        if val_size > 0 and inp.shape[0] > val_size * 2.5:
            if verbose:
                print("created validation from %s" % (dt))
            k = random.sample(np.arange(inp.shape[0]).tolist(), val_size)
            val_src.append(inp[k, :, :])
            val_trg.append(out[k, :, :])
            val_seq_start.append(dt_seq_start[k, :, :])
            val_frames.append(dt_frames[k, :])
            val_dt.append(dt_dataset[k])
            val_peds.append(dt_peds[k])
            inp = np.delete(inp, k, 0)
            out = np.delete(out, k, 0)
            dt_frames = np.delete(dt_frames, k, 0)
            dt_seq_start = np.delete(dt_seq_start, k, 0)
            dt_dataset = np.delete(dt_dataset, k, 0)
            dt_peds = np.delete(dt_peds, k, 0)
        elif val_size > 0:
            if verbose:
                print("could not create validation from %s, size -> %i" %
                      (dt, inp.shape[0]))

        data_src.append(inp)  # source
        data_trg.append(out)  # target
        data_seq_start.append(dt_seq_start)
        data_frames.append(dt_frames)
        data_dt.append(dt_dataset)
        data_peds.append(dt_peds)

    data['src'] = np.concatenate(data_src, 0)
    data['trg'] = np.concatenate(data_trg, 0)
    data['seq_start'] = np.concatenate(data_seq_start, 0)
    data['frames'] = np.concatenate(data_frames, 0)
    data['dataset'] = np.concatenate(data_dt, 0)
    data['peds'] = np.concatenate(data_peds, 0)
    data['dataset_name'] = datasets_list

    mean = data['src'].mean((0, 1))
    std = data['src'].std((0, 1))

    if val_size > 0:
        # data_val = {}
        # data_val['src'] = np.concatenate(val_src, 0)
        # data_val['trg'] = np.concatenate(val_trg, 0)
        # data_val['seq_start'] = np.concatenate(val_seq_start, 0)
        # data_val['frames'] = np.concatenate(val_frames, 0)
        # data_val['dataset'] = np.concatenate(val_dt, 0)
        # data_val['peds'] = np.concatenate(val_peds, 0)
        data_val = {
            'src': np.concatenate(val_src, 0),
            'trg': np.concatenate(val_trg, 0),
            'seq_start': np.concatenate(val_seq_start, 0),
            'frames': np.concatenate(val_frames, 0),
            'dataset': np.concatenate(val_dt, 0),
            'peds': np.concatenate(val_peds, 0)
        }

        return IndividualTfDataset(data, "train", mean, std), IndividualTfDataset(data_val, "validation", mean, std)

    return IndividualTfDataset(data, "train", mean, std), None

    # return IndividualTfDataset(data, "train", mean, std), IndividualTfDataset(data_val, "validation", mean, std)


def create_old_3dim_dataset(dataset_folder,
                            dataset_name,
                            val_size=0,
                            gt=8,
                            horizon=0,
                            delim='',  # 该数据集场景下使用默认的split函数即可
                            train=True,
                            eval=False,
                            verbose=True):
    """
    从 train、val等子文件夹下加载数据，适用于侯师兄的数据集版本
    """
    os.chdir(PROJECT_FOLDER)
    file_list = None  # 雷达数据集文件列表
    full_dt_folder = None
    dataset_type = None
    if train:
        dataset_type = 'train'
        file_list = os.listdir(os.path.join(
            dataset_folder, dataset_name, dataset_type))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "train")
    if not train and not eval:
        # cerate validation dataset
        dataset_type = 'val'
        file_list = os.listdir(os.path.join(
            dataset_folder, dataset_name, dataset_type))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, 'val')
    if not train and eval:
        dataset_type = 'test'
        file_list = os.listdir(os.path.join(
            dataset_folder, dataset_name, dataset_type))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, 'test')

    data = {}
    data_times = []
    data_dt = []

    # 雷达数据集
    launch_points = []
    curves = []

    if verbose:
        print("\033[0;33mStart loading %s dataset \033[0m" % dataset_type)

    file_list_with_index = [(i, j) for i, j in enumerate(file_list)]

    for i_dt, dt in tqdm(file_list_with_index):
        # 脏数据处理
        if 'data' not in dt:
            continue
        # 总数据量约为16003，分割后训练集约6400
        # if verbose and ((i_dt + 1) % 1000 or i_dt + 1 == len(file_list)):
        #     print("\033[0;31m %03i / %03i \033[0m loading %s " %
        #           (i_dt + 1, len(file_list), dt))
        with open(os.path.join(full_dt_folder, dt), 'r') as f:
            launch_point = map(float, f.readline().strip())
            radar_position = map(float, f.readline().strip())
            curve = []  # 完整的轨迹
            for i, line in enumerate(f):
                if i > 399:
                    break
                strs = line.strip().split()  # 注意不需要使用其他的分割符号
                tmp = np.array(list(map(float, strs))).astype(np.float)
                curve.append(tmp)
            curve = np.stack(curve)
            launch_points.append(curve[0])
            curves.append(curve)
            # curves中包含的是完整轨迹，此时需要考虑点的间隔数
            # 由于采样了400个点，再采样时设置间隔为10
            # 可以将20个点当做训练集，20个点当做GT
            # train_set = np.array(curve[200:400:10]).astype(np.float)
            # target_set = np.array(curve[0:200:10]).astype(np.float)
            sample = np.stack(curve[0:400:10]).astype(float)  # shape [40, 4]

            d_time = np.concatenate(sample[:, [3]])  # 时间维度
            # inp_speed = np.concatenate(
            #     np.zeros((sample.shape[0] - 1, 1, 3)),
            #     np.array(sample[1:, :, [0, 1, 2]] - sample[:-1, :, [0, 1, 2]])
            # )
            data_times.append(d_time)
            d_data = np.stack(sample[:, 0:3])  # 一个轨迹采样
            # 轨迹随机平移处理
            shift = [np.random.rand() * 50, np.random.rand() * 50]
            d_data[:, :2] += shift
            data_dt.append(d_data)

    data_times = np.stack(data_times)
    data_dt_np = np.stack(data_dt)
    data_dt_np = data_dt_np[:, ::-1, :]
    inp_speed = np.concatenate(
        (np.zeros((data_dt_np.shape[0], 1, 3)),
         data_dt_np[:, 1:, :] - data_dt_np[:, :-1, :]),
        1)

    inp_norm = np.concatenate((data_dt_np, inp_speed), 2)
    inp_mean = np.zeros(6)
    inp_std = np.ones(6)

    # # TODO 注意，由于是外推操作，应该将轨迹翻转
    # FIXME 这个翻转要放到上面执行，这样才能产生正确的预测结果
    inp = inp_norm[:, 0:20, :]  # 使用copy解决出现负维度问题
    out = inp_norm[:, 20:, :]

    data['src'] = inp
    data['trg'] = out
    data['time'] = data_times

    mean = data['src'].mean((0, 1))
    std = data['src'].std((0, 1))

    return IndividualTf3DDataset(data, "train", mean=mean, std=std), None


def create_new_3dim_dataset(dataset_folder,
                            dataset_name,
                            val_size,
                            gt,
                            horizon,
                            delim='\t',
                            train=True,
                            eval=False,
                            verbose=True):
    """
    新数据集加载函数，可通过 gt 和 horizon 自定义观测点数和预测点数。

    :param gt 观测点数
    :param horizon 外推点数
    """
    os.chdir(PROJECT_FOLDER)

    # 雷达文件列表
    file_list = os.listdir(os.path.join(dataset_folder, dataset_name))
    full_dt_folder = os.path.join(dataset_folder, dataset_name)

    data_train = {}
    data_val = {}
    data_test = {}
    data_dt = []

    # 取样点数
    sample_number = gt + horizon 

    # 雷达数据集
    launch_points = []
    curves = []

    if verbose:
        print("\033[0;33mStart loading all data \033[0m")

    file_list_with_index = [(i, j) for i, j in enumerate(file_list)]

    for i_dt, dt in tqdm(file_list_with_index):
        # 脏数据处理
        if 'data' not in dt:
            continue
        with open(os.path.join(full_dt_folder, dt), 'r') as f:
            launch_point = map(float, f.readline().strip())
            radar_position = map(float, f.readline().strip())
            curve = []  # 记录一条轨迹
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
            margin = curve.shape[0] // (sample_number)
            sample = np.stack(curve[::margin]).astype(float)[0:(sample_number)]
            assert sample.shape[0] == (sample_number)

            d_data = np.stack(sample[:, 0:3])  # 一个轨迹采样
            # 轨迹随机平移
            # shift = [np.random.rand() * 50, np.random.rand() * 50]
            # d_data[:, :2] += shift

            data_dt.append(d_data)

    data_dt_np = np.stack(data_dt)
    # NOTE: 对抽取到的轨迹做翻转
    data_dt_np = data_dt_np[:, ::-1, :]  
    inp_speed = np.concatenate(
        (np.zeros((data_dt_np.shape[0], 1, 3)),
         data_dt_np[:, 1:, :] - data_dt_np[:, :-1, :]), 1)

    inp_norm = np.concatenate((data_dt_np, inp_speed), 2)
    inp_mean = np.zeros(6)
    inp_std = np.ones(6)

    # NOTE: 逆序轨迹，先取雷达观测到的部分，再取 GT
    inp = inp_norm[:, 0:horizon, :]
    out = inp_norm[:, horizon:, :]

    # 数据集分割
    inp_train, inp_val_test, out_train, out_val_test = train_test_split(inp, out, test_size=0.2, shuffle=True)
    inp_val, inp_test, out_val, out_test = train_test_split(inp_val_test, out_val_test, test_size=0.5, shuffle=True)

    data_train['src'] = inp_train
    data_train['trg'] = out_train

    data_val['src'] = inp_val
    data_val['trg'] = out_val

    data_test['src'] = inp_test
    data_test['trg'] = out_test

    mean_train = data_train['src'].mean((0, 1))
    std_train = data_train['src'].std((0, 1))

    mean_val = data_val['src'].mean((0, 1))
    std_val = data_val['src'].std((0, 1))

    mean_test = data_test['src'].mean((0, 1))
    std_test = data_test['src'].std((0, 1))

    return IndividualTf3DDataset(data_train, "train", mean=mean_train, std=std_train), \
           IndividualTf3DDataset(data_val, "val", mean=mean_val, std=std_val), \
           IndividualTf3DDataset(data_test, "test", mean=mean_test, std=std_test)


class IndividualTf3DDataset(Dataset):
    """
    自定义三维数据集
    """

    def __init__(self, data, name, mean, std):
        super(IndividualTf3DDataset, self).__init__()

        self.data = data
        self.name = name
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.data['src'].shape[0]

    def __getitem__(self, index):
        return {
            'src': torch.Tensor(self.data['src'][index]),
            'trg': torch.Tensor(self.data['trg'][index]),
        }


class IndividualTfDataset(Dataset):
    def __init__(self, data: dict, name: str, mean, std):
        super(IndividualTfDataset, self).__init__()

        self.data = data
        self.name = name

        self.mean = mean
        self.std = std

    def __len__(self):
        return self.data['src'].shape[0]

    def __getitem__(self, index):
        return {'src': torch.Tensor(self.data['src'][index]),
                'trg': torch.Tensor(self.data['trg'][index]),
                'frames': self.data['frames'][index],
                'seq_start': self.data['seq_start'][index],
                'dataset': self.data['dataset'][index],  # 数据集的编号
                'peds': self.data['peds'][index],
                }


def create_folders(baseFolder, datasetName):
    try:
        os.mkdir(baseFolder)
    except:
        pass

    try:
        os.mkdir(os.path.join(baseFolder, datasetName))
    except:
        pass


def get_strided_data(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame = []
    ped_ids = []
    for p in ped:
        for i in range(1 + (raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i *
                                                       step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i *
                                                           step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames = np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids = np.stack(ped_ids)

    inp_no_start = inp_te_np[:, 1:, 0:2] - inp_te_np[:, :-1, 0:2]
    inp_std = inp_no_start.std(axis=(0, 1))
    inp_mean = inp_no_start.mean(axis=(0, 1))
    inp_norm = inp_no_start
    # inp_norm = (inp_no_start - inp_mean) / inp_std

    # vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    # inp_norm=np.concatenate((inp_norm,vis),2)

    return inp_norm[:, :gt_size - 1], inp_norm[:, gt_size - 1:], {'mean': inp_mean, 'std': inp_std,
                                                                  'seq_start': inp_te_np[:, 0:1, :].copy(),
                                                                  'frames': frames, 'peds': ped_ids}


def get_strided_data_2(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame = []
    ped_ids = []
    for p in ped:
        for i in range(1 + (raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i *
                                                       step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i *
                                                           step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames = np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids = np.stack(ped_ids)

    inp_relative_pos = inp_te_np - inp_te_np[:, :1, :]
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0], 1, 2)), inp_te_np[:, 1:, 0:2] - inp_te_np[:, :-1, 0:2]),
                               1)
    inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0], 1, 2)), inp_speed[:, 1:, 0:2] - inp_speed[:, :-1, 0:2]),
                               1)
    # inp_std = inp_no_start.std(axis=(0, 1))
    # inp_mean = inp_no_start.mean(axis=(0, 1))
    # inp_norm= inp_no_start
    # inp_norm = (inp_no_start - inp_mean) / inp_std

    # vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    # inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm = np.concatenate(
        (inp_te_np, inp_relative_pos, inp_speed, inp_accel), 2)
    inp_mean = np.zeros(8)
    inp_std = np.ones(8)

    return inp_norm[:, :gt_size], inp_norm[:, gt_size:], {'mean': inp_mean, 'std': inp_std,
                                                          'seq_start': inp_te_np[:, 0:1, :].copy(), 'frames': frames,
                                                          'peds': ped_ids}


def get_strided_data_clust(dt: DataFrame, gt_size, horizon, step):
    """
    获取步幅数据集群
    """
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    # 获取行人的唯一编号
    ped = raw_data.ped.unique()
    # frame 可以当做时间戳
    frame = []
    ped_ids = []
    for p in ped:
        for i in range(1 + (raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            # iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze()
            # 获取 i * step : i * step + gt_size + horizon 行的第一列数据
            # 这里的第一列数据就是帧号
            frame.append(dt[dt.ped == p].iloc[i * step:i *
                                                       step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i *
                                                           step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames = np.stack(frame)  # 默认是将第 0 维聚合起来，生成一个 numpy 对象
    inp_te_np = np.stack(inp_te)
    ped_ids = np.stack(ped_ids)

    # inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    # 此处是用下一个位置的值减去当前位置的值，由于相差不大，因此结果都是处于 0 ～ 1 之间的
    inp_speed = np.concatenate(
        (np.zeros((inp_te_np.shape[0], 1, 2)),  # 将 [0. , 0. ]放到每个轨迹的最前头
         inp_te_np[:, 1:, 0:2] - inp_te_np[:, :-1, 0:2]), 1)
    # inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    # inp_std = inp_no_start.std(axis=(0, 1))
    # inp_mean = inp_no_start.mean(axis=(0, 1))
    # inp_norm= inp_no_start
    # inp_norm = (inp_no_start - inp_mean) / inp_std

    # vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    # inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm = np.concatenate((inp_te_np, inp_speed), 2)
    inp_mean = np.zeros(4)
    inp_std = np.ones(4)

    # 注意最后输出的依然是一个 numpy 对象
    return inp_norm[:, :gt_size], inp_norm[:, gt_size:], {
        'mean': inp_mean,
        'std': inp_std,
        'seq_start': inp_te_np[:, 0:1, :].copy(),
        'frames': frames,
        'peds': ped_ids
    }


def distance_metrics(gt, preds):
    """
    计算欧氏距离，三维和二维通用
    """
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(
                gt[i, j], preds[i, j])
    return errors.mean(), errors[:, -1].mean(), errors


def distance_metrics_3dims(gt, preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(
                gt[i, j], preds[i, j])
    return errors.mean(), errors[:, -1].mean(), errors


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot
