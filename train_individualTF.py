import argparse
import os

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import baselineUtils
from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt


def main():
    parser = argparse.ArgumentParser(description='Train the individual Transformer model | 训练独立Transformer模型')
    parser.add_argument('--dataset_folder', type=str, default='datasets')
    parser.add_argument('--dataset_name', type=str, default='zara1')
    parser.add_argument('--obs', type=int, default=8)  # TODO
    parser.add_argument('--preds', type=int, default=12)  # TODO
    parser.add_argument('--emb_size', type=int, default=512)  # 编码层大小
    parser.add_argument('--heads', type=int, default=8)  # TODO    
    parser.add_argument('--layers', type=int, default=6)  # TODO
    parser.add_argument('--dropout', type=float, default=0.1)  # DropOut 默认值
    parser.add_argument('--cpu', action='store_true')  # 设置是否使用CPU
    parser.add_argument('--val_size', type=int, default=0)  # 验证集的尺寸
    parser.add_argument('--verbose', action='store_true')  # 是否详细输出日志
    parser.add_argument('--max_epoch', type=int, default=1500)  # 最大训练
    parser.add_argument('--batch_size', type=int, default=70)  # 批大小
    parser.add_argument('--validation_epoch_start', type=int, default=30)  # TODO 可能是验证次代起始位置？
    parser.add_argument('--resume_train', action='store_true')  # TODO 继续训练？
    parser.add_argument('--delim', type=str, default='\t')  # 标识数据集的分隔符
    parser.add_argument('--name', type=str, default="zara1")  # 模型名称
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # 当设置为使用CPU并且GPU不可用的时候才会使用CPU进行训练
    if args.cpu or not torch.cuda.is_available():
        print("\033[1;31;40m Training with cpu... \033[0m")
        device = torch.device("cpu")

    args.verbose = True

    ## creation of the dataloaders for train and validation
    # 创建训练集和验证集的 dataloader
    if args.val_size == 0:
        train_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs, args.preds,
                                                        delim=args.delim, train=True, verbose=args.verbose)
        val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs,
                                                      args.preds, delim=args.delim, train=False,
                                                      verbose=args.verbose)
    else:
        train_dataset, val_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,
                                                                  args.obs,
                                                                  args.preds, delim=args.delim, train=True,
                                                                  verbose=args.verbose)
    # 测试集创建
    test_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs, args.preds,
                                                   delim=args.delim, train=False, eval=True, verbose=args.verbose)

    import individual_TF
    model = individual_TF.IndividualTF(2, 3, 3, N=args.layers,
                                       d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,
                                       mean=[0, 0], std=[0, 0]).to(device)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)
    validate_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=0)

    # optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    # sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    # 使用了自己实现的 Optimizer
    optim = NoamOpt(args.emb_size, args.factor, len(train_dataloader) * args.warmup,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch = 0

    # mean=train_dataset[:]['src'][:,1:,2:4].mean((0,1))
    mean = torch.cat((train_dataset[:]['src'][:, 1:, 2:4], train_dataset[:]['trg'][:, :, 2:4]), 1).mean((0, 1))
    # std=train_dataset[:]['src'][:,1:,2:4].std((0,1))
    std = torch.cat((train_dataset[:]['src'][:, 1:, 2:4], train_dataset[:]['trg'][:, :, 2:4]), 1).std((0, 1))
    means = []
    stds = []
    for i in np.unique(train_dataset[:]['dataset']):
        ind = train_dataset[:]['dataset'] == i
        means.append(
            torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
        stds.append(
            torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)

    scipy.io.savemat(f'models/Individual/{args.name}/norm.mat', {'mean': mean.cpu().numpy(), 'std': std.cpu().numpy()})

    print("\033[1;31;40m Start Training... \033[0m")

    while epoch < args.max_epoch:
        epoch_loss = 0
        model.train()

        for id_b, batch in enumerate(train_dataloader):
            optim.optimizer.zero_grad()
            # (batch_size, 7, 2)
            inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
            # (batch_size, 11, 2)
            target = (batch['trg'][:, :-1, 2:4].to(device) - mean.to(device)) / std.to(device)
            target_c = torch.zeros((target.shape[0], target.shape[1], 1)).to(device)
            # 第三维合并
            target = torch.cat((target, target_c), -1)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0], 1, 1).to(device)

            # (batch_size, 1, 3)
            decoder_input = torch.cat((start_of_seq, target), 1)

            # (input_shape[0], 1, input_shape[1] | 7)
            src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            # (batch_size, 1, 1)
            trg_att = subsequent_mask(decoder_input.shape[1]).repeat(decoder_input.shape[0], 1, 1).to(device)

            # (batch_size, 12, 3)
            prediction = model(inp, decoder_input, src_att, trg_att)

            # 计算两个矩阵的成对距离
            loss = F.pairwise_distance(prediction[:, :, 0:2].contiguous().view(-1, 2),
                                       ((batch['trg'][:, :, 2:4].to(device) - mean.to(device)) / std.to(
                                           device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(
                torch.abs(prediction[:, :, 2]))
            loss.backward()
            optim.step()
            print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (
                epoch, args.max_epoch, id_b, len(train_dataloader), loss.item()))
            epoch_loss += loss.item()
        # sched.step()
        log.add_scalar('Loss/train', epoch_loss / len(train_dataloader), epoch)

        print("\033[1;31;40m Start Validating... \033[0m")

        with torch.no_grad():
            model.eval()

            val_loss = 0
            step = 0
            model.eval()
            gt = []
            pr = []
            input_ = []
            peds = []
            frames = []
            dt = []

            for id_b, batch in enumerate(validate_dataloader):
                input_.append(batch['src'])
                gt.append(batch['trg'][:, :, 0:2])
                frames.append(batch['frames'])
                peds.append(batch['peds'])
                dt.append(batch['dataset'])

                inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
                src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                    device)
                decoder_input = start_of_seq

                for i in range(args.preds):
                    trg_att = subsequent_mask(decoder_input.shape[1]).repeat(decoder_input.shape[0], 1, 1).to(device)
                    out = model(inp, decoder_input, src_att, trg_att)
                    decoder_input = torch.cat((decoder_input, out[:, -1:, :]), 1)

                preds_tr_b = (decoder_input[:, 1:, 0:2] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + \
                             batch['src'][:, -1:, 0:2].cpu().numpy()
                pr.append(preds_tr_b)
                print("val epoch %03i/%03i  batch %04i / %04i" % (
                    epoch, args.max_epoch, id_b, len(validate_dataloader)))

            peds = np.concatenate(peds, 0)
            frames = np.concatenate(frames, 0)
            dt = np.concatenate(dt, 0)
            gt = np.concatenate(gt, 0)
            dt_names = test_dataset.data['dataset_name']
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
            log.add_scalar('validation/MAD', mad, epoch)
            log.add_scalar('validation/FAD', fad, epoch)

            if args.evaluate:

                model.eval()
                gt = []
                pr = []
                input_ = []
                peds = []
                frames = []
                dt = []

                for id_b, batch in enumerate(test_dataloader):
                    input_.append(batch['src'])
                    gt.append(batch['trg'][:, :, 0:2])
                    frames.append(batch['frames'])
                    peds.append(batch['peds'])
                    dt.append(batch['dataset'])

                    inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
                    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                        device)
                    decoder_input = start_of_seq

                    for i in range(args.preds):
                        trg_att = subsequent_mask(decoder_input.shape[1]).repeat(decoder_input.shape[0], 1, 1).to(
                            device)
                        out = model(inp, decoder_input, src_att, trg_att)
                        decoder_input = torch.cat((decoder_input, out[:, -1:, :]), 1)

                    preds_tr_b = (decoder_input[:, 1:, 0:2] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(
                        1) + \
                                 batch['src'][:, -1:, 0:2].cpu().numpy()
                    pr.append(preds_tr_b)
                    print("test epoch %03i/%03i  batch %04i / %04i" % (
                        epoch, args.max_epoch, id_b, len(test_dataloader)))

                peds = np.concatenate(peds, 0)
                frames = np.concatenate(frames, 0)
                dt = np.concatenate(dt, 0)
                gt = np.concatenate(gt, 0)
                dt_names = test_dataset.data['dataset_name']
                pr = np.concatenate(pr, 0)
                mad, fad, errs = baselineUtils.distance_metrics(gt, pr)

                log.add_scalar('eval/DET_mad', mad, epoch)
                log.add_scalar('eval/DET_fad', fad, epoch)

                # log.add_scalar('eval/DET_mad', mad, epoch)
                # log.add_scalar('eval/DET_fad', fad, epoch)

                scipy.io.savemat(f"output/Individual/{args.name}/det_{epoch}.mat",
                                 {'input': inp, 'gt': gt, 'pr': pr, 'peds': peds, 'frames': frames, 'dt': dt,
                                  'dt_names': dt_names})

        if epoch % args.save_step == 0:
            torch.save(model.state_dict(), f'models/Individual/{args.name}/{epoch:05d}.pth')

        epoch += 1


if __name__ == '__main__':
    main()
