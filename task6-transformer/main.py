import math
import os
from typing import Union, Dict, Tuple, Optional, Sequence, List, Any
import torch
from optuna import Trial
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl

CKPT_PATH = "newest.ckpt"


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


def positional_encoding(x: Tensor):
    """
    为输入添加位置编码。
    :param x: (batch_size,sentence_len,vector_dim)
    """
    batch_size, sentence_len, vector_dim = x.shape
    position = torch.range(0, sentence_len - 1)  # (sentence_len)
    position = torch.unsqueeze(position, dim=1).expand(sentence_len, vector_dim)  # (sentence_len,vector_dim)
    division = torch.exp(torch.range(0, vector_dim - 1, 2) * -math.log(1e4) / vector_dim)  # (vector_dim // 2)
    division = torch.unsqueeze(division, dim=1).expand(sentence_len, vector_dim // 2)  # (sentence_len,vector_dim // 2)
    pe = torch.zeros(sentence_len, vector_dim)
    pe[:, 0::2] = torch.sin(position[:, 0::2] * division)
    pe[:, 1::2] = torch.cos(position[:, 1::2] * division)
    return x + pe


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
    """
    注：有关 vocab_id 的约定：
    0：padding，即无文字也无标志；
    1 ~ vocab_size-3：普通的文字；
    vocab_size-2：<BOS>；
    vocab_size-1：<EOS>。

    数据集在给数据时，始终应该：x 两端无任何标志。y 两端有 <BOS> 和 <EOS> 标志。
    """

    @staticmethod
    def create_from_trial(trial: Trial, vocab_size: int):
        return Transformer(stack_num=trial.suggest_int("stack_num", 1, 8), vocab_size=vocab_size,
                           embed_dim=trial.suggest_int("embed_dim", 128, 1024, 8),
                           ffn_hidden_dim=trial.suggest_int("ffn_hidden_dim", 512, 2048, 8),
                           head_num=trial.suggest_categorical("head_num", [1, 2, 4, 8]),
                           dropout=trial.suggest_float("dropout", 0.1, 0.5))

    def __init__(self, stack_num: int, vocab_size: int, embed_dim: int, ffn_hidden_dim: int, head_num: int,
                 dropout: float, *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)

        default_biased = False
        self.head_num = head_num
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoders = nn.ModuleList(
            [EncoderLayer(embed_dim, ffn_hidden_dim, head_num, dropout) for _ in range(stack_num)])
        self.decoders = nn.ModuleList(
            [DecoderLayer(embed_dim, embed_dim, ffn_hidden_dim, head_num, dropout) for _ in range(stack_num)])
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, vocab_size, bias=default_biased),
            nn.Softmax(dim=-1)
        )

    def core_calculate(self, x: PackedSequence, y: PackedSequence):
        x_max_len, y_max_len = torch.max(x.batch_sizes), torch.max(y.batch_sizes)
        max_len = max(x_max_len, y_max_len)

        x, x_lens = pad_packed_sequence(x, batch_first=True, total_length=max_len)
        y, y_lens = pad_packed_sequence(y, batch_first=True, total_length=max_len)
        # 需保证训练时看到的 y 不含 <EOS>。
        y_lens -= 1

        embedded_x, embedded_y = \
            self.embedding(x) * math.sqrt(self.embed_dim), \
            self.embedding(y) * math.sqrt(self.embed_dim)  # (batch_size,max_len,embed_dim)
        embedded_x, embedded_y = positional_encoding(embedded_x), positional_encoding(embedded_y)

        for i in range(self.head_num):
            embedded_x = self.encoders[i](embedded_x, x_lens)
        for i in range(self.head_num):
            embedded_y = self.decoders[i](embedded_y, embedded_x, y_lens, x_lens)

        return self.output_layer(embedded_y)  # (batch_size,max_len,vocab_size)

    def training_step(self, batch: Tuple[PackedSequence, PackedSequence], batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        x, y = batch
        output = self.core_calculate(x, y)  # (batch_size,max_len,vocab_size)
        batch_size, max_len, _ = output.shape
        output = torch.flatten(output, start_dim=0, end_dim=1)  # (batch_size*max_len,vocab_size)
        y, _ = pad_packed_sequence(y, batch_first=True, total_length=max_len)  # (batch_size,max_len)
        # 删去第一个字符（<EOS>），并在最后补充一列
        y = torch.cat([y[:, 1:], torch.zeros((batch_size, 1))], dim=-1)
        y = torch.flatten(y)

        loss = F.cross_entropy(output, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Optional[Union[
        Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]
    ]]:
        return torch.optim.Adam(self.parameters(), lr=1 / math.sqrt(self.embed_dim))

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.trainer.save_checkpoint(CKPT_PATH)


def main():
    pass


if __name__ == '__main__':
    main()