import multiprocessing
import os
from typing import Union, Dict, Tuple, Optional, Sequence, List, Any, T_co

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl

CKPT_PATH = "newest.ckpt.2"


def elementwise_apply(fn, packed_sequence: PackedSequence):
    """applies a point wise function fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)


class CharRNN(pl.LightningModule):

    def __init__(self, vocab_size: int, vector_size: int, rnn_hidden_size: int, use_lstm: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, vector_size)
        if use_lstm:
            self.rnn = nn.LSTM(vector_size, rnn_hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(vector_size, rnn_hidden_size, batch_first=True)
        self.smp = nn.Sequential(nn.Dropout(), nn.Linear(rnn_hidden_size, vocab_size), nn.Softmax(dim=-1))

    def forward(self, x: str, dataset: "TangPoetryDataset") -> Tensor:
        """

        :param dataset: 数据集
        :param x: 文本
        :return 下一个字
        """
        sentence = torch.tensor([dataset.word2idx[char] for char in x[-dataset.word_bag_size:]], device=self.device,
                                dtype=torch.int64)
        x = self.embedding(sentence)  # (word_len,vector_size)
        x, _ = self.rnn(x)  # (word_len,rnn_hidden_size)
        x = self.smp(x)  # (word_len,vocab_size)

        return dataset.idx2word[torch.argmax(x[-1]).item()]

    def training_step(self, batch: Tuple[PackedSequence, PackedSequence], batch_idx) -> Union[
        int, Dict[str, Union[Tensor, Dict[str, Tensor]]]
    ]:

        x, y = batch
        x = elementwise_apply(self.embedding, x)  # (batch_size,word_len,vector_size)
        x, _ = self.rnn(x)  # (batch_size,word_len,rnn_hidden_size)
        x = elementwise_apply(self.smp, x)  # (batch_size,word_len,vocab_size)
        # x = elementwise_apply(lambda e: torch.argmax(e,dim=-1), x)  # (batch_size,word_len)
        x, _ = pad_packed_sequence(x, batch_first=True)
        y, _ = pad_packed_sequence(y, batch_first=True)
        x, y = torch.flatten(x, 0, 1), torch.flatten(y, 0, 1)
        loss = F.cross_entropy(x, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Optional[Union[
        Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]
    ]]:
        return torch.optim.RMSprop(self.parameters(), lr=1e-3)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.trainer.save_checkpoint(CKPT_PATH)


class TangPoetryDataset(Dataset):
    def __init__(self, file_path: str, word_bag_size: int):
        text, self.word2idx, self.idx2word, self.vocab_size = self.read_dataset(file_path)
        self.word_bag_size = word_bag_size
        self.x, self.y = self.generate_vectors(text, word_bag_size)

    def __getitem__(self, index: Any) -> T_co:
        if isinstance(index, list):
            return [self.x[i] for i in index], [self.y[i] for i in index],
        return self.x[index], self.y[index]

    def read_dataset(self, file_name):
        with open(file_name, 'r', encoding="utf-8") as file:
            text = file.read().strip().replace("\n", "")
            vocabs = sorted(list(set(text)))
            word2idx = dict((c, i) for i, c in enumerate(vocabs))
            idx2word = dict((i, c) for i, c in enumerate(vocabs))
            return text, word2idx, idx2word, len(vocabs)

    def generate_vectors(self, text: str, word_bag_size: int) -> Tuple[
        List[Tensor], List[Tensor]]:
        features = []
        next_features = []
        for i in range(len(text) - word_bag_size):
            sentence = text[i:i + word_bag_size]
            feature = torch.zeros((word_bag_size,), dtype=torch.int64)
            for j, char in enumerate(sentence):
                feature[j] = self.word2idx[char]
            features.append(feature)

            sentence = text[i + 1:i + word_bag_size + 1]
            label = torch.zeros((word_bag_size,), dtype=torch.int64)
            for j, char in enumerate(sentence):
                label[j] = self.word2idx[char]
            next_features.append(label)

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


def main():
    VECTOR_SIZE = 600
    HIDDEN_SIZE = 300
    EPOCHES = 1
    train_dataset = TangPoetryDataset("poetryFromTang.txt", 20)
    if os.path.exists(CKPT_PATH):
        model = CharRNN.load_from_checkpoint(CKPT_PATH,
                                             vocab_size=train_dataset.vocab_size, vector_size=VECTOR_SIZE,
                                             rnn_hidden_size=HIDDEN_SIZE)
    else:
        model = CharRNN(vocab_size=train_dataset.vocab_size, vector_size=VECTOR_SIZE, rnn_hidden_size=HIDDEN_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pack_collate_fn
                              )
    trainer = pl.Trainer(max_epochs=EPOCHES, accelerator="gpu")
    trainer.fit(model=model, train_dataloaders=train_loader)

    def predict(initial_text: str, length: int):
        print(initial_text, end='')
        for i in range(length):
            new_char = model(initial_text, train_dataset)
            initial_text += new_char
            print(new_char, end='')

    predict("何幸遇休明", 100)


if __name__ == '__main__':
    main()
