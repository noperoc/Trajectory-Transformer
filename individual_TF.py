import copy
import math

import numpy as np
import torch.nn as nn

from transformer.decoder import Decoder
from transformer.decoder_layer import DecoderLayer
from transformer.encoder import Encoder
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder_layer import EncoderLayer
from transformer.multihead_attention import MultiHeadAttention
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.positional_encoding import PositionalEncoding


class IndividualTF(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, N=6,
                 d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]):
        """
        :param enc_inp_size Encoder Input Size
        :param dec_inp_size Decoder Input Size
        :param dec_out_size Decoder Output Size
        :param N 网络层数
        """
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        # 深拷贝alias
        deepcopy = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)  # 前馈神经网络
        position = PositionalEncoding(d_model, dropout)  # 进行位置编码
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N),
            Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn),
                                 deepcopy(ff), dropout), N),
            nn.Sequential(LinearEmbedding(enc_inp_size, d_model), deepcopy(position)),
            nn.Sequential(LinearEmbedding(dec_inp_size, d_model), deepcopy(position)),
            Generator(d_model, dec_out_size))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, *input):
        return self.model.generator(self.model(*input))


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)
