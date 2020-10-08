# -*- coding: utf-8 -*-
# date: 2018-11-29 20:14
import torch

import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    https://zhuanlan.zhihu.com/p/33173246
    横向规范化，用同一个规范化操作转换不同纬度的输入。
    """

    def __init__(self, features, eps=1e-6):
        """
        :param features 特征维度
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
