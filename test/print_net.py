# 打印网络的输出

import argparse
import os

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import baselineUtils
from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt


def main():
    parser = argparse.ArgumentParser(description='Train the individual Transformer model | 训练独立Transformer模型')
    parser.add_argument('--dataset_folder', type=str, default='datasets')
    parser.add_argument('--dataset_name', type=str, default='radar')
    parser.add_argument('--obs', type=int, default=20)  # TODO 观测点，单条训练数据的长度
    parser.add_argument('--preds', type=int, default=20)  # TODO 预测点数，要设置为数据集的大小
    parser.add_argument('--emb_size', type=int, default=512)  # 编码层大小
    parser.add_argument('--heads', type=int, default=8)  # TODO
    parser.add_argument('--layers', type=int, default=6)  # TODO
    parser.add_argument('--dropout', type=float, default=0.1)  # DropOut 默认值
    parser.add_argument('--cpu', action='store_true')  # 设置是否使用CPU
    parser.add_argument('--val_size', type=int, default=0)  # 验证集的尺寸
    parser.add_argument('--verbose', action='store_true')  # 是否详细输出日志
    parser.add_argument('--max_epoch', type=int, default=500)  # TODO 最大训练次代
    parser.add_argument('--batch_size', type=int, default=70)  # 批大小
    parser.add_argument('--validation_epoch_start', type=int, default=30)  # TODO 可能是验证次代起始位置？
    parser.add_argument('--resume_train', action='store_true')  # TODO 继续训练？
    parser.add_argument('--delim', type=str, default='\t')  # 标识数据集的分隔符
    parser.add_argument('--name', type=str, default="radar")  # 模型名称
    parser.add_argument('--factor', type=float, default=1.)  # TODO
    parser.add_argument('--save_step', type=int, default=1)  # TODO
    parser.add_argument('--warmup', type=int, default=10)  # TODO 开始热身的批次
    parser.add_argument('--evaluate', type=bool, default=True)  # 是否对数据集进行评估

    args = parser.parse_args()
    model_name = args.name

    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/Individual')
    except:
        pass
    try:
        os.mkdir(f'models/Individual')
    except:
        pass

    try:
        os.mkdir(f'output/Individual/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/Individual/{args.name}')
    except:
        pass

    log = SummaryWriter('logs/Ind_%s' % model_name)

    log.add_scalar('eval/mad', 0, 0)
    log.add_scalar('eval/fad', 0, 0)
    device = torch.device("cuda")

    # 当设置为使用CPU并且GPU不可用的时候才会使用CPU进行训练
    if args.cpu or not torch.cuda.is_available():
        print("\033[1;31;40m Training with cpu... \033[0m")
        device = torch.device("cpu")

    args.verbose = True

    ## creation of the dataloaders for train and validation
    # 创建训练集和验证集的 dataloader
    train_dataset, _ = baselineUtils.create_old_3dim_dataset(
        args.dataset_folder,
        args.dataset_name,
        0,
        args.obs,
        args.preds,
        train=True,
        verbose=args.verbose)

    import individual_TF_3D
    model = individual_TF_3D.IndividualTF(3, 4, 4, N=args.layers,
                                          d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,
                                          mean=[0, 0, 0], std=[0, 0, 0]).to(device)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)

    # optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    # sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    # 使用了自己实现的 Optimizer
    # FIXME betas 参数是否需要继续调整
    optim = NoamOpt(args.emb_size, args.factor, len(train_dataloader) * args.warmup,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch = 0

    # mean=train_dataset[:]['src'][:,1:,2:4].mean((0,1))
    # mean = torch.cat((train_dataset[:]['src'][:, 1:, 2:4], train_dataset[:]['trg'][:, :, 2:4]), 1).mean((0, 1))
    # std=train_dataset[:]['src'][:,1:,2:4].std((0,1))
    # std = torch.cat((train_dataset[:]['src'][:, 1:, 2:4], train_dataset[:]['trg'][:, :, 2:4]), 1).std((0, 1))
    # means = []
    # stds = []
    # for i in np.unique(train_dataset[:]['dataset']):
    #     ind = train_dataset[:]['dataset'] == i
    #     means.append(
    #         torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
    #     stds.append(
    #         torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
    mean = torch.cat((train_dataset[:]['src'][:, 1:, 3:6], train_dataset[:]['trg'][:, :, 3:6]), 1).mean((0, 1))
    std = torch.cat((train_dataset[:]['src'][:, 1:, 3:6], train_dataset[:]['trg'][:, :, 3:6]), 1).std((0, 1))

    scipy.io.savemat(f'models/Individual/{args.name}/norm.mat', {'mean': mean.cpu().numpy(), 'std': std.cpu().numpy()})

    model.train()

    train_batch_bar = tqdm([(id, batch) for id, batch in enumerate(train_dataloader)][0:1])

    prediction = None

    for id_b, batch in train_batch_bar:
        optim.optimizer.zero_grad()
        # (batch_size, 19, 3)
        inp = (batch['src'][:, 1:, 3:6].to(device) - mean.to(device)) / std.to(device)
        # (batch_size, 11, 2)
        target = (batch['trg'][:, :-1, 3:6].to(device) - mean.to(device)) / std.to(device)
        target_c = torch.zeros((target.shape[0], target.shape[1], 1)).to(device)
        # 第三维合并
        target = torch.cat((target, target_c), -1)
        start_of_seq = torch.Tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0], 1, 1).to(device)

        # (batch_size, 20, 4)
        decoder_input = torch.cat((start_of_seq, target), 1)

        # (input_shape[0], 1, input_shape[1] | 19)
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
        # (batch_size, 20, 20)
        trg_att = subsequent_mask(decoder_input.shape[1]).repeat(decoder_input.shape[0], 1, 1).to(device)

        # (batch_size, 20, 4)
        prediction = model(inp, decoder_input, src_att, trg_att)

        # 计算两个矩阵的成对距离
        loss = F.pairwise_distance(prediction[:, :, 0:3].contiguous().view(-1, 2),
                                   ((batch['trg'][:, :, 3:6].to(device) - mean.to(device)) / std.to(
                                       device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(
            torch.abs(prediction[:, :, 3]))
        loss.backward()
        optim.step()
        train_batch_bar.set_description("loss: %7.4f" % loss.item())
        # print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (
        #     epoch, args.max_epoch, id_b, len(train_dataloader), loss.item()))

    graph = baselineUtils.make_dot(prediction)
    graph.view()
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))


if __name__ == '__main__':
    main()
