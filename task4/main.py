import collections
from math import floor
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import nn, Tensor
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, random_split
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


def sum_tensor_list(values):
    summation = torch.zeros_like(values[0])
    for i in range(0, len(values)):
        summation += values[i]
    return summation


class CRF(nn.Module):
    def __init__(self, tag_num):
        super().__init__()
        self.n = tag_num
        self.A = nn.Parameter(torch.empty((tag_num + 2, tag_num + 2)))
        nn.init.uniform_(self.A, -0.1, 0.1)

    def expand_P(self, P: Tensor, real_lengths: Tensor) -> Tensor:
        """
        在 P 矩阵两端添加 START 和 END 标记。
        :param P: (batch_size,word_len,tag_num)
        :param real_lengths: (batch_size)
        :return: 扩展后的 P(batch_size,word_len+2,tag_num+2)
        """
        batch_size, word_len, tag_num = P.shape
        small = -1000
        P = torch.cat([torch.ones(batch_size, 1, tag_num) * small, P,
                       torch.ones(batch_size, 1, tag_num) * small], dim=1)  # (batch_size,word_len+2,tag_num)
        for i in range(batch_size):
            P[i][real_lengths[i] + 1] = torch.ones(tag_num) * small

        only_first_is_one = torch.cat(
            [torch.ones((batch_size, 1, 1)), torch.ones((batch_size, word_len + 1, 1)) * small], dim=1)
        only_last_is_one = torch.ones((batch_size, word_len + 2, 1)) * small
        for i in range(batch_size):
            only_last_is_one[i][real_lengths[i] + 1][0] = 1
        return torch.cat([P, only_first_is_one, only_last_is_one], dim=2)  # (batch_size,word_len+2,tag_num+2)

    def expand_prediction(self, prediction: Tensor, real_length: int):
        """
        在 Prediction 两端添加 START 和 END 标记。
        :param prediction: (word_len)
        :param real_length:
        :return: 扩展后的 Prediction(word_len+2)
        """
        expanded_prediction = torch.cat(
            [torch.ones((1,), dtype=torch.int64) * self.n, prediction,
             torch.ones((1,), dtype=torch.int64) * (self.n + 1)],
            dim=-1)  # (word_len+2)
        expanded_prediction[real_length + 1] = self.n + 1
        return expanded_prediction

    def _score(self, prediction: Tensor, P: Tensor, real_length: int):
        """
        计算 Prediction 的分数。
        :param prediction: (word_len)
        :param P: (word_len+2,tag_num+2)
        :return: (1)
        """
        prediction = self.expand_prediction(prediction, real_length)

        s1 = sum_tensor_list([self.A[prediction[i], prediction[i + 1]] for i in range(real_length + 1)])
        s2 = sum_tensor_list([P[i, prediction[i]] for i in range(1, real_length + 1)])
        return s1 + s2

    def _Z(self, P: Tensor, real_length: int) -> Tensor:
        """
        计算所有可能的 Prediction 的分数之和。
        :param P: (word_len+2,tag_num+2)
        :return: (1)
        """
        expP = torch.exp(P)
        expA = torch.exp(self.A)
        expsubA = expA[:self.n, :self.n]  # (n,n)
        # 第 0 步的选择
        Z = expA[[self.n], :self.n] * expP[[1], :self.n]  # (1,n)
        # 递推在所有路径上求积（指数的和）
        for t in range(2, real_length + 1):
            Z = (Z @ expsubA) * expP[[t], :self.n]  # (1,n)
        result = torch.squeeze(torch.log(torch.sum(Z * expA[:self.n, self.n + 1], dim=1)), dim=-1)
        return result

    def viterbi(self, P: Tensor, word_len: int, tag_num: int):
        """
        用最长路算法，找出对应的极大似然标注。
        :param tag_num
        :param word_len
        :param P: (word_len+2,tag_num+2)
        :return: (word_len)
        """
        min_length = torch.zeros((word_len, tag_num))
        last_step = -1 * torch.ones((word_len, tag_num), dtype=torch.int32)
        # 第 0 步
        min_length[0] = self.A[[self.n], :self.n] + P[1, :tag_num]
        # 动态规划求下一步
        for t in range(1, word_len):
            for this_index in range(tag_num):
                current_best_length = -10000
                current_best_index = -1
                for last_index in range(tag_num):
                    edge_weight = self.A[last_index][this_index] + P[t + 1][this_index]
                    if min_length[t - 1][last_index] + edge_weight > current_best_length:
                        current_best_index = last_index
                        current_best_length = min_length[t - 1][last_index] + edge_weight
                min_length[t][this_index] = current_best_length
                last_step[t][this_index] = current_best_index

        # 收束的最后一步
        total_best_length = -10000
        total_best_index = -1
        for last_index in range(tag_num):
            edge_weight = self.A[last_index][self.n + 1]
            if min_length[-1, last_index] + edge_weight > total_best_length:
                total_best_length = min_length[-1, last_index] + edge_weight
                total_best_index = last_index
        path = []
        while total_best_index >= 0:
            path.append(total_best_index)
            total_best_index = last_step[word_len - len(path)][total_best_index].item()
        path.reverse()
        return path

    def predict(self, P: Tensor, real_lengths: Tensor):
        """
        根据输入预测结果。实质上是接受 Batch 形式的 viterbi() 的包装。
        :param real_lengths:  (batch_size)
        :param P: (batch_size,word_len+2,tag_num+2)
        :return: (batch_size,max_word_len)
        """
        batch_size, _, tag_num_plus_two = P.shape
        return rnn.pad_sequence(
            [torch.tensor(self.viterbi(P[i], real_lengths[i].item(), tag_num_plus_two - 2), dtype=torch.int64) for i in
             range(batch_size)],
            batch_first=True, padding_value=-1)

    def forward(self, y: Tensor, P: Tensor, real_lengths: Tensor, output_loss=True):
        """

        :param y: 正确的标注(batch_size,word_len)。预测时可赋 None。
        :param P: (batch_size,word_len,tag_num)
        :param real_lengths: (batch_size)
        :param output_loss: CRF 层作为损失函数还是预测函数。
        :return:
        """
        batch_size, word_len, _ = P.shape
        P = self.expand_P(P, real_lengths)
        if output_loss:
            return torch.stack(
                [self._Z(P[i], real_lengths[i]) - self._score(y[i], P[i], real_lengths[i]) for i in range(batch_size)])
        else:
            padded_result = self.predict(P, real_lengths)
            return F.pad(padded_result, (0, word_len - padded_result.shape[1]), value=-1)


class LSTMCRF(nn.Module):
    def __init__(self, dict_size, word_num, vector_size, lstm_hidden_size, tag_num):
        super(LSTMCRF, self).__init__()
        self.embedding = nn.Embedding(dict_size, vector_size, padding_idx=-1)
        self.input_encoding = \
            nn.LSTM(vector_size, lstm_hidden_size, batch_first=True, bidirectional=True, bias=False)
        self.smp = nn.Sequential(nn.BatchNorm1d(word_num),
                                 nn.Linear(2 * lstm_hidden_size, tag_num, bias=False), nn.Softmax(dim=-1))
        self.crf_layer = CRF(tag_num)

    def forward(self, x: Tensor, real_lengths: Tensor, y: Tensor, output_loss=True):
        # 输入 x (batch_size,word_num), y (batch_size,word_num)
        x = self.embedding(x)  # (batch_size,word_num,vector_size)
        x, _ = self.input_encoding(x)  # (batch_size,word_num,2 * lstm_hidden_size)
        x = self.smp(x)  # (batch_size,word_num,tag_num)
        return self.crf_layer(y, x, real_lengths, output_loss=output_loss)


class CONLLDataset(Dataset):
    def __init__(self, file_path):
        x_data, y_data = self.read_dataset(file_path)
        self.x, self.x_lens, self.y, self._dict_size = self.generate_vectors(x_data, y_data)
        self.x, self.x_lens, self.y = map(lambda x: torch.from_numpy(x).to(device), (self.x, self.x_lens, self.y))

    def __getitem__(self, index: Any) -> T_co:
        return self.x[index], self.x_lens[index], self.y[index]

    def word_num(self):
        return self.x.shape[-1]

    def dict_size(self):
        return self._dict_size

    def read_dataset(self, file_name):
        x_data = []
        x_line = []
        y_data = []
        y_line = []
        with open(file_name, 'r') as file:
            for line in file:
                parts = line.strip().split(" ")
                if len(parts) != 4:
                    if len(x_line) > 0 and len(y_line) > 0:
                        x_data.append(x_line.copy())
                        y_data.append(y_line.copy())
                        x_line.clear()
                        y_line.clear()
                else:
                    x_line.append(parts[0])
                    y_line.append(parts[3])
        if len(x_line) > 0 and len(y_line) > 0:
            x_data.append(x_line.copy())
            y_data.append(y_line.copy())
        return x_data, y_data

    def generate_vectors(self, texts: List[List[str]], labels: List[List[str]]) -> Tuple[
        ndarray, ndarray, ndarray, int]:
        phrase_dict: Dict[str, int] = {}
        real_lengths = np.zeros((len(texts)), dtype=np.int64)
        max_text_len = 0
        for i, sentence in enumerate(texts):
            max_text_len = max(max_text_len, len(sentence))
            real_lengths[i] = len(sentence)
            for phrase in sentence:
                if phrase not in phrase_dict:
                    phrase_dict[phrase] = len(phrase_dict)
        features = np.zeros((len(texts), max_text_len), dtype=np.int64)
        for i, sentence in enumerate(texts):
            for j, phrase in enumerate(sentence):
                features[i][j] = phrase_dict[phrase]
        num_labels = np.ones((len(texts), max_text_len), dtype=np.int64) * -1
        for i, label in enumerate(labels):
            for j, single_label in enumerate(label):
                num_labels[i][j] = \
                    {'O': 0, 'I-ORG': 1, 'I-PER': 2, 'I-LOC': 3, 'I-MISC': 4, 'B-MISC': 5, 'B-LOC': 6, 'B-ORG': 7,
                     'B-PER': 8}[
                        single_label]
        return features, real_lengths, num_labels, len(phrase_dict)

    def __len__(self):
        return len(self.x_lens)


def get_labels(subset):
    return [(torch.count_nonzero(torch.where(subset[i][2] > 0)[0]) > subset[i][1] / 2).int().item() for i in
            range(len(subset))]


def evaluate(model: nn.Module, x: Tensor, x_lens: Tensor, y: Tensor):
    batch_size = x.shape[0]
    result = model(x, x_lens, None, output_loss=False).int()
    evaluation = []
    zeros = []
    zeros_y = []
    for i in range(batch_size):
        evaluation.append(
            torch.sum(torch.eq(result[i][:x_lens[i].item()], y[i][:x_lens[i].item()]).int()) / x_lens[i])
        zeros.append(torch.sum(torch.eq(result[i][:x_lens[i].item()], torch.zeros(x_lens[i].item())).int()))
        zeros_y.append(torch.sum(torch.eq(y[i][:x_lens[i].item()], torch.zeros(x_lens[i].item())).int()))

    return {"validation_accu": sum_tensor_list(evaluation) / batch_size,
            "zero": sum_tensor_list(zeros),
            "zero_y": sum_tensor_list(zeros_y)}


def main():
    # Torch 训练优化
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)

    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
    dataset = CONLLDataset("eng.testa")
    per = floor(0.8 * len(dataset))
    train_dataset, valid_dataset = random_split(dataset, (per, len(dataset) - per), generator=torch.Generator(device))
    model = LSTMCRF(dataset.dict_size(), dataset.word_num(), 300, 100, 8).to(device)

    batch_size = 8
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=get_labels))
    loss_history = []
    for epoch in range(100):
        loop = tqdm(total=len(train_loader))
        for i, data in enumerate(train_loader):
            loss = model(*data)
            pred = evaluate(model, *data)
            # 增加惩罚项，迫使网络不要输出 0
            loss = loss.mean() + torch.relu((pred["zero"] - pred["zero_y"])) * 100
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            loop.set_description(f'Epoch [{epoch}/100]')
            loop.set_postfix(loss=sum(loss_history[-10:]) / len(loss_history[-10:]),
                             **evaluate(model, *valid_dataset[:10]))
            loop.update()
        loop.close()
        torch.save(model.state_dict(), "result.pt")


if __name__ == '__main__':
    main()
