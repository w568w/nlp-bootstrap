import os
from typing import Union, Dict, Tuple, Optional, Sequence, List, Any
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl


def build_mask_from_padding_lengths(real_lengths: Tensor, padded_length: int):
    """
    根据长度构建遮罩矩阵。
    如：real_lengths = [1 3], padded_length = 4，
    则返回两个矩阵：
    F T T T
    F T T T
    F T T T
    F T T T
    和
    F F F T
    F F F T
    F F F T
    F F F T
    :param real_lengths: (batch_size)
    :param padded_length: pad 后的句子长度
    :return: (batch_size,padded_length,padded_length)
    """
    batch_size = real_lengths.shape[0]
    mask = torch.full((batch_size, padded_length, padded_length), False)
    for i in range(batch_size):
        mask[i, :, real_lengths[i].item():] = True
    return mask


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    执行缩放点积注意力计算。
    :param query: (batch_size,sentence_len,key_dim)
    :param key: (batch_size,sentence_len,key_dim)
    :param value: (batch_size,sentence_len,value_dim)
    :param mask: 遮罩层
    """
    _, sentence_len, key_dim = query.shape
    attention = (query @ key.T) / torch.sqrt(key_dim)  # (batch_size,sentence_len,sentence_len)
    if mask:
        attention = torch.masked_fill(attention, mask, -1e100)
    attention = torch.softmax(attention, dim=-1)
    return attention @ value  # (batch_size,sentence_len,value_dim)


def add_and_norm_dropout(module: nn.Module, input_tensor: Tensor, module_dropout: Optional[float] = None, *module_args):
    if module_dropout:
        result = input_tensor + F.dropout(module(*module_args), module_dropout)
    else:
        result = input_tensor + module(*module_args)
    return F.layer_norm(result, result.shape[-1])


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num: int, in_key_dim: int, in_value_dim: int, hidden_key_dim: int, hidden_value_dim: int,
                 out_dim: Optional[int] = None):
        super().__init__()
        if out_dim is None:
            out_dim = in_key_dim
        default_biased = False
        self.head_num = head_num
        self.pre_key_projections = nn.ModuleList(
            [nn.Linear(in_key_dim, hidden_key_dim, bias=default_biased) for _ in range(head_num)])
        self.pre_query_projections = nn.ModuleList(
            [nn.Linear(in_key_dim, hidden_key_dim, bias=default_biased) for _ in range(head_num)])
        self.pre_value_projections = nn.ModuleList(
            [nn.Linear(in_value_dim, hidden_value_dim, bias=default_biased) for _ in range(head_num)])

        self.post_output_projection = nn.Linear(head_num * hidden_value_dim, out_dim, bias=default_biased)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, padding_mask: Optional[Tensor],
                upper_tri_masked: bool = False):
        """
        :param query:
        :param key:
        :param value:
        :param padding_mask: (sentence_len, sentence_len)
        :param upper_tri_masked: 是否添加上三角遮罩。通常是为了避免 Decoder 看到不必要的后置信息
        """
        batch_size, sentence_len, _ = query.shape

        if upper_tri_masked:
            upper_mask = torch.triu(torch.full((sentence_len, sentence_len), True), diagonal=1)
            # 将 upper_mask 复制 batch_size 份，变成 (batch_size,sentence_len,sentence_len)
            upper_mask = torch.unsqueeze(upper_mask, dim=0).expand([batch_size, sentence_len, sentence_len])
            padding_mask = torch.bitwise_or(upper_mask, padding_mask)  # (batch_size,sentence_len,sentence_len)

        heads: List[Tensor] = []
        for i in range(self.head_num):
            heads.append(
                scaled_dot_product_attention(self.pre_query_projections[i](query),
                                             self.pre_key_projections[i](key),
                                             self.pre_value_projections[i](value),
                                             mask=padding_mask))
        aggregate_head = torch.cat(heads, dim=-1)
        return self.post_output_projection(aggregate_head)  # (batch_size,sentence_len,out_dim)


class PointWiseFFN(nn.Module):
    """
    简单的两层感知器。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.in_layer(x)


class EncoderLayer(nn.Module):
    def __init__(self, in_out_dim: int, ffn_hidden_dim: int, head_num: int, sublayer_dropout: float):
        super().__init__()
        if in_out_dim % head_num != 0:
            raise ValueError("Unable to construct encoder layer.")
        multi_head_split_dim = in_out_dim // head_num
        self.sublayer_dropout = sublayer_dropout
        self.multi_head_attention = MultiHeadAttention(head_num, in_out_dim, in_out_dim, multi_head_split_dim,
                                                       multi_head_split_dim, in_out_dim)
        self.ffn = PointWiseFFN(in_out_dim, ffn_hidden_dim, in_out_dim)

    def forward(self, x: Tensor, real_lengths: Tensor):
        """
        :param real_lengths: (batch_size)
        :param x: (batch_size,sentence_len,in_out_dim)
        """
        dropout = self.sublayer_dropout if self.training else None
        sentence_len = x.shape[-2]
        x = add_and_norm_dropout(self.multi_head_attention, x, dropout,
                                 (x, x, x, build_mask_from_padding_lengths(
                                     real_lengths, sentence_len)))  # (batch_size,sentence_len,in_out_dim)
        return add_and_norm_dropout(self.ffn, x, dropout,
                                    (x,))  # (batch_size,sentence_len,in_out_dim)


class DecoderLayer(nn.Module):
    def __init__(self, encoder_out_dim: int, in_out_dim: int, ffn_hidden_dim: int, head_num: int,
                 sublayer_dropout: float):
        super().__init__()
        if in_out_dim % head_num != 0 or encoder_out_dim % head_num != 0:
            raise ValueError("Unable to construct decoder layer.")
        multi_head_split_dim = in_out_dim // head_num
        encoder_multi_head_split_dim = encoder_out_dim // head_num
        self.sublayer_dropout = sublayer_dropout
        self.masked_multi_head_attention = MultiHeadAttention(head_num, in_out_dim, in_out_dim, multi_head_split_dim,
                                                              multi_head_split_dim, in_out_dim)
        self.multi_head_attention = MultiHeadAttention(head_num, encoder_out_dim, in_out_dim,
                                                       encoder_multi_head_split_dim, multi_head_split_dim, in_out_dim)
        self.ffn = PointWiseFFN(in_out_dim, ffn_hidden_dim, in_out_dim)

    def forward(self, x: Tensor, encoder_output: Tensor, real_lengths: Tensor, encoder_real_lengths: Tensor):
        """
        :param encoder_real_lengths: (batch_size)
        :param encoder_output: (batch_size,sentence_len,encoder_out_dim)
        :param real_lengths: (batch_size)
        :param x: (batch_size,sentence_len,in_out_dim)
        注意：输入给 decoder 的每个句子 x[i]，应在最前面添加一个开始标记。或者说，所有句子都要右移一位；
        注意：对 encoder 和 decoder 的每个 batch 的输入，应当补齐到相同的 sentence_len。
        """
        dropout = self.sublayer_dropout if self.training else None
        sentence_len = x.shape[-2]
        x = add_and_norm_dropout(self.masked_multi_head_attention, x, dropout,
                                 (x, x, x, build_mask_from_padding_lengths(
                                     real_lengths, sentence_len), True))
        x = add_and_norm_dropout(self.multi_head_attention, x, dropout,
                                 (encoder_output, encoder_output, x,
                                  build_mask_from_padding_lengths(encoder_real_lengths, sentence_len)))
        return add_and_norm_dropout(self.ffn, x, dropout,
                                    (x,))  # (batch_size,sentence_len,in_out_dim)


class Transformer(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
