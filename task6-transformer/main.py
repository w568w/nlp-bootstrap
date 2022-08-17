import math
from typing import Union, Dict, Tuple, Optional, Sequence, List, Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from optuna import Trial
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, Dataset

CKPT_PATH = "newest.ckpt"


def build_mask_from_padding_lengths(real_lengths: Tensor, padded_length: int, device: Device):
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
    :param device: 向量所用的设备
    :param real_lengths: (batch_size)
    :param padded_length: pad 后的句子长度
    :return: (batch_size,padded_length,padded_length)
    """
    batch_size = real_lengths.shape[0]
    real_lengths = real_lengths.cpu()
    mask = torch.full((batch_size, padded_length, padded_length), False, device=device)
    for i in range(batch_size):
        mask[i, :, real_lengths[i]:] = True
    return mask


def select_header_as_batch(t: Tensor, last_index: Tensor) -> List[Tensor]:
    """
    对于 i = 0 ~ batch_size-1，选择 t[i] 的前 last_index[i] 项，拼成数组。
    :param t: (batch_size,...)
    :param last_index: (batch_size)
    """
    return [t[i, :last_index[i], ...] for i in range(t.shape[0])]


def add_and_norm_dropout(module: nn.Module, input_tensor: Tensor, module_args, module_dropout: Optional[float] = None):
    if module_dropout:
        result = input_tensor + F.dropout(module(*module_args), module_dropout)
    else:
        result = input_tensor + module(*module_args)

    return F.layer_norm(result, [result.shape[-1]])


def positional_encoding(x: Tensor, device: Device):
    """
    为输入添加位置编码。
    :param device: 向量所在的设备
    :param x: (batch_size,sentence_len,vector_dim)
    """
    batch_size, sentence_len, vector_dim = x.shape
    position = torch.arange(sentence_len, device=device, dtype=torch.float)  # (sentence_len)
    position = torch.unsqueeze(position, dim=1).expand(sentence_len, vector_dim)  # (sentence_len,vector_dim)
    division = torch.exp(
        torch.arange(0, vector_dim, 2, device=device, dtype=torch.float) * -math.log(
            1e4) / vector_dim)  # (vector_dim // 2)
    division = torch.unsqueeze(division, dim=0).expand(sentence_len, vector_dim // 2)  # (sentence_len,vector_dim // 2)
    pe = torch.zeros(sentence_len, vector_dim, device=device)
    pe[:, 0::2] = torch.sin(position[:, 0::2] * division)
    pe[:, 1::2] = torch.cos(position[:, 1::2] * division)
    return x + pe


class ScaledDotProductAttention(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        执行缩放点积注意力计算。
        :param query: (batch_size,sentence_len,key_dim)
        :param key: (batch_size,sentence_len,key_dim)
        :param value: (batch_size,sentence_len,value_dim)
        :param mask: 遮罩层
        """
        _, sentence_len, key_dim = query.shape
        attention = query @ torch.transpose(key, -1, - 2) / math.sqrt(key_dim)  # (batch_size,sentence_len,sentence_len)
        if mask is not None:
            attention.masked_fill_(mask, -1e8)
        attention_softed = torch.softmax(attention, dim=-1)
        return attention_softed @ value  # (batch_size,sentence_len,value_dim)


class MultiHeadAttention(pl.LightningModule):
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
        self.inner_attention_layer = ScaledDotProductAttention()

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
            upper_mask = torch.full((sentence_len, sentence_len), True, device=self.device).triu_(diagonal=1)
            padding_mask.logical_or_(upper_mask)  # (batch_size,sentence_len,sentence_len)

        heads: List[Tensor] = []
        for i in range(self.head_num):
            heads.append(
                self.inner_attention_layer(self.pre_query_projections[i](query),
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


class EncoderLayer(pl.LightningModule):
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
        x = add_and_norm_dropout(self.multi_head_attention, x,
                                 (x, x, x, build_mask_from_padding_lengths(
                                     real_lengths, sentence_len, self.device)),
                                 dropout)  # (batch_size,sentence_len,in_out_dim)
        return add_and_norm_dropout(self.ffn, x,
                                    (x,), dropout)  # (batch_size,sentence_len,in_out_dim)


class DecoderLayer(pl.LightningModule):
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
        x = add_and_norm_dropout(self.masked_multi_head_attention, x,
                                 (x, x, x, build_mask_from_padding_lengths(
                                     real_lengths, sentence_len, self.device), True), dropout)
        x = add_and_norm_dropout(self.multi_head_attention, x,
                                 (encoder_output, encoder_output, x,
                                  build_mask_from_padding_lengths(encoder_real_lengths, sentence_len, self.device)),
                                 dropout)
        return add_and_norm_dropout(self.ffn, x, (x,), dropout)  # (batch_size,sentence_len,in_out_dim)


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
        return Transformer(stack_num=trial.suggest_int("stack_num", 1, 2), vocab_size=vocab_size,
                           embed_dim=trial.suggest_int("embed_dim", 128, 1024, 8),
                           ffn_hidden_dim=trial.suggest_int("ffn_hidden_dim", 512, 2048, 8),
                           head_num=trial.suggest_categorical("head_num", [1, 2, 4]),
                           dropout=trial.suggest_float("dropout", 0.1, 0.5))

    def __init__(self, stack_num: int, vocab_size: int, embed_dim: int, ffn_hidden_dim: int, head_num: int,
                 dropout: float, *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)

        default_biased = False
        self.head_num = head_num
        self.stack_num = stack_num
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
        max_len = max(len(x.batch_sizes), len(y.batch_sizes))

        x, x_lens = pad_packed_sequence(x, batch_first=True, total_length=max_len)
        y, y_lens = pad_packed_sequence(y, batch_first=True, total_length=max_len)
        # 需保证训练时看到的 y 不含 <EOS>。
        y_lens -= 1

        embedded_x, embedded_y = \
            self.embedding(x) * math.sqrt(self.embed_dim), \
            self.embedding(y) * math.sqrt(self.embed_dim)  # (batch_size,max_len,embed_dim)
        embedded_x, embedded_y = positional_encoding(embedded_x, self.device), positional_encoding(embedded_y,
                                                                                                   self.device)

        for i in range(self.stack_num):
            embedded_x = self.encoders[i](embedded_x, x_lens)
        for i in range(self.stack_num):
            embedded_y = self.decoders[i](embedded_y, embedded_x, y_lens, x_lens)

        return self.output_layer(embedded_y)  # (batch_size,max_len,vocab_size)

    def forward(self, input: Tensor, known_output: Tensor, *args, **kwargs) -> int:
        """

        :param input: 输入句子 (sentence_len)
        :param known_output: 已知的输出部分 (known_length <= sentence_len)
        :return:
        """
        known_length = known_output.shape[0]
        x = pack_sequence_cpu([input])
        # 末尾随意补充一个元素
        y = pack_sequence_cpu([F.pad(known_output, [0, 1])])
        result = self.core_calculate(x, y)  # (1,max_sentence_len,vocab)
        return torch.argmax(result[0][known_length], dim=-1).item()

    def training_step(self, batch: Tuple[PackedSequence, PackedSequence], batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        x, y = batch
        output = self.core_calculate(x, y)  # (batch_size,max_len,vocab_size)
        batch_size, max_len, _ = output.shape

        y, y_lens = pad_packed_sequence(y, batch_first=True, total_length=max_len)  # (batch_size,max_len)
        # 删去第一个字符（<BOS>）
        y = y[:, 1:]
        y_lens -= 1

        loss = F.cross_entropy(torch.cat(select_header_as_batch(output, y_lens)),
                               torch.cat(select_header_as_batch(y, y_lens)))
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self) -> Optional[Union[
        Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]
    ]]:
        return torch.optim.Adam(self.parameters(), lr=1 / math.sqrt(self.embed_dim))

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.trainer.save_checkpoint(CKPT_PATH)


class QQMessageDataset(Dataset):
    def __init__(self, file_path: str):
        texts, self.word2idx, self.idx2word, self.vocab_size = self.read_dataset(file_path)
        self.x, self.y = self.generate_vectors(texts)

    def __getitem__(self, index: Any):
        if isinstance(index, list):
            return [self.x[i] for i in index], [self.y[i] for i in index],
        return self.x[index], self.y[index]

    @staticmethod
    def read_dataset(file_name):
        with open(file_name, 'r', encoding="utf-8") as file:
            texts = list(filter(lambda x: len(x) <= 100, file.read().strip().split("\n===\n")))
            vocabs = sorted(list(set(''.join(texts))))
            word2idx = dict((c, i + 1) for i, c in enumerate(vocabs))
            word_nums = len(vocabs)
            word2idx["<BOS>"] = word_nums
            word2idx["<EOS>"] = word_nums + 1
            idx2word = dict((i + 1, c) for i, c in enumerate(vocabs))
            idx2word[0] = "<NULL>"
            idx2word[word_nums] = "<BOS>"
            idx2word[word_nums + 1] = "<EOS>"
            return texts, word2idx, idx2word, word_nums + 2

    def generate_vectors(self, texts: List[str]) -> Tuple[List[Tensor], List[Tensor]]:
        features = []
        next_features = []
        for i in range(len(texts) - 1):
            feature = torch.tensor([self.word2idx[word] for word in texts[i]], dtype=torch.int64)
            next_feature = [self.word2idx["<BOS>"]]
            next_feature.extend([self.word2idx[word] for word in texts[i + 1]])
            next_feature.append(self.word2idx["<EOS>"])
            next_feature = torch.tensor(next_feature, dtype=torch.long)
            features.append(feature)
            next_features.append(next_feature)
        return features, next_features

    def __len__(self):
        return len(self.x)


def pack_sequence_cpu(sequences, enforce_sorted=True):
    lengths = torch.as_tensor([v.size(0) for v in sequences], device=torch.device("cpu"))
    return pack_padded_sequence(pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted)


def pack_collate_fn(data):
    x = list(map(lambda t: t[0], data))
    y = list(map(lambda t: t[1], data))
    return pack_two(x, y)


def pack_two(x, y):
    x.sort(key=lambda x: len(x), reverse=True)
    y.sort(key=lambda x: len(x), reverse=True)
    return pack_sequence_cpu(x), pack_sequence_cpu(y)


EPOCHS = 10


def main():
    train_dataset = QQMessageDataset("data.txt")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pack_collate_fn, pin_memory=True)

    # def objective(trial):
    #     model = Transformer.create_from_trial(trial, train_dataset.vocab_size)
    #
    #     trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="cpu")
    #     trainer.fit(model=model, train_dataloaders=train_loader)
    #     return trainer.logged_metrics["train_loss"].item()  # An objective value linked with the Trial object.
    #
    # study = optuna.create_study()  # Create a new study.
    # study.optimize(objective, n_trials=1)  # Invoke optimization of the objective function.
    model: Transformer
    if False:
        model = Transformer.load_from_checkpoint(CKPT_PATH, stack_num=8, vocab_size=train_dataset.vocab_size,
                                                 embed_dim=512,
                                                 ffn_hidden_dim=1024, head_num=8, dropout=0.1)
    else:
        model = Transformer(stack_num=8, vocab_size=train_dataset.vocab_size,
                            embed_dim=512,
                            ffn_hidden_dim=1024, head_num=8, dropout=0.1)

    def predict(model: Transformer, initial_text: str):
        initial_text = initial_text.strip()
        initial_text = torch.tensor([train_dataset.word2idx[i] for i in initial_text], dtype=torch.int64)
        output = [train_dataset.word2idx["<BOS>"]]
        while train_dataset.idx2word[output[-1]] != "<EOS>":
            next_word: int = model(initial_text, torch.tensor(output, dtype=torch.int64))
            output.append(next_word)
            print(train_dataset.idx2word[next_word], end='')

    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="gpu")
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()
