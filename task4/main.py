import collections
from typing import List, Tuple, Dict

import numpy as np
import pandas
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import nn, Tensor
from tqdm import tqdm


def sum_nonempty(values):
    summation = values[0]
    for i in range(1, len(values)):
        summation += values[i]
    return summation


class CRF(nn.Module):
    def __init__(self, tag_num):
        super().__init__()
        self.n = tag_num
        self.A = nn.Parameter(torch.empty((tag_num + 2, tag_num + 2)))
        nn.init.uniform_(self.A, -0.1, 0.1)

    def expand_P(self, P: Tensor) -> Tensor:
        batch_size, word_len, tag_num = P.shape
        small = -1000
        P = torch.cat([torch.ones(batch_size, 1, tag_num) * small, P,
                       torch.ones(batch_size, 1, tag_num) * small], dim=1)  # (batch_size,word_len+2,tag_num)
        only_first_is_one = torch.cat(
            [torch.ones((batch_size, 1, 1)), torch.ones((batch_size, word_len + 1, 1)) * small], dim=1)
        only_last_is_one = torch.cat(
            [torch.ones((batch_size, word_len + 1, 1)) * small, torch.ones((batch_size, 1, 1))], dim=1)
        return torch.cat([P, only_first_is_one, only_last_is_one], dim=2)  # (batch_size,word_len+2,tag_num+2)

    def score(self, prediction: Tensor, P: Tensor):
        """

        :param prediction: (batch_size,word_len)
        :param P: (batch_size,word_len,tag_num)
        :return: (batch_size)
        """
        batch_size, word_len = prediction.shape

        # 分别拼接 prediction 和 P 的两侧，使之达到应有的 shape
        prediction = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.int64) * self.n, prediction,
             torch.ones((batch_size, 1), dtype=torch.int64) * (self.n + 1)],
            dim=-1)  # (batch_size,word_len+2)
        P = self.expand_P(P)

        s1 = sum_nonempty([self.A[prediction[..., i], prediction[..., i + 1]] for i in range(word_len + 1)])
        s2 = torch.Tensor(
            [sum_nonempty([P[j, i, prediction[j, i]] for i in range(1, word_len + 1)]) for j in range(0, batch_size)])
        return s1 + s2

    def Z(self, P: Tensor) -> Tensor:
        """

        :param P: (batch_size,word_len,tag_num)
        :return: (batch_size)
        """
        batch_size, word_len, _ = P.shape
        P = self.expand_P(P)
        expP = torch.exp(P)
        expA = torch.exp(self.A)
        expsubA = expA[:self.n, :self.n]  # (n,n)
        # 第 0 步的选择
        Z = expA[[self.n], :self.n] * expP[:, [1], :self.n]  # (batch_size,1,n)
        # 递推在所有路径上求积（指数的和）
        for t in range(2, word_len + 1):
            Z = (Z @ expsubA) * expP[:, [t], :self.n]  # (batch_size,1,n)
        result = torch.squeeze(torch.log(torch.sum(Z * expA[:self.n, self.n + 1], dim=2)), dim=-1)
        return result

    def viterbi(self, P: Tensor, word_len: int, tag_num: int):
        """

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
                current_best_length = 1e5
                current_best_index = -1
                for last_index in range(tag_num):
                    edge_weight = self.A[last_index][this_index] + P[t + 1][this_index]
                    if min_length[t - 1][last_index] + edge_weight < current_best_length:
                        current_best_index = last_index
                        current_best_length = min_length[t - 1][last_index] + edge_weight
                min_length[t][this_index] = current_best_length
                last_step[t][this_index] = current_best_index

        # 收束的最后一步
        total_best_length = 1e5
        total_best_index = -1
        for last_index in range(tag_num):
            edge_weight = self.A[last_index][self.n + 1]
            if min_length[-1, last_index] + edge_weight < total_best_length:
                total_best_length = min_length[-1, last_index] + edge_weight
                total_best_index = last_index
        path = []
        while total_best_index >= 0:
            path.append(total_best_index)
            total_best_index = last_step[word_len - len(path)][total_best_index].item()
        path.reverse()
        return path

    def predict(self, P: Tensor):
        """

        :param P: (batch_size,word_len,tag_num)
        :return: (batch_size,word_len)
        """
        batch_size, word_len, tag_num = P.shape
        P = self.expand_P(P)
        return torch.tensor([self.viterbi(P[i], word_len, tag_num) for i in range(batch_size)])

    def forward(self, y: Tensor, P: Tensor, output_loss=True):
        if output_loss:
            return -(self.score(y, P) - self.Z(P))
        else:
            return self.predict(P)


class LSTMCRF(nn.Module):
    def __init__(self, dict_size, vector_size, lstm_hidden_size, tag_num):
        super(LSTMCRF, self).__init__()
        self.embedding = nn.Embedding(dict_size, vector_size, padding_idx=0)
        self.input_encoding = \
            nn.LSTM(vector_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.smp = nn.Sequential(nn.Linear(2 * lstm_hidden_size, tag_num), nn.Softmax(dim=-1))
        self.crf_layer = CRF(tag_num)

    def forward(self, x, y, output_loss=True):
        # 输入 x (batch_size,word_num), y (batch_size,word_num)
        x = self.embedding(x)  # (batch_size,word_num,vector_size)
        x, _ = self.input_encoding(x)  # (batch_size,word_num,2 * lstm_hidden_size)
        x = self.smp(x)  # (batch_size,word_num,tag_num)
        return self.crf_layer(y, x, output_loss=output_loss)


def read_dataset(file_name):
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


def generate_vectors(texts: List[List[str]], labels: List[List[str]]) -> Tuple[ndarray, ndarray, int]:
    phrase_dict: Dict[str, int] = {}
    max_text_len = 0
    for sentence in texts:
        max_text_len = max(max_text_len, len(sentence))
        for phrase in sentence:
            if phrase not in phrase_dict:
                phrase_dict[phrase] = len(phrase_dict)
    features = np.zeros((len(texts), max_text_len), dtype=np.int64)
    for i, sentence in enumerate(texts):
        for j, phrase in enumerate(sentence):
            features[i][j] = phrase_dict[phrase]
    num_labels = np.zeros((len(texts), max_text_len), dtype=np.int64)
    for i, label in enumerate(labels):
        for j, single_label in enumerate(label):
            num_labels[i][j] = {'O': 0, 'I-ORG': 1, 'I-PER': 2, 'I-LOC': 3, 'I-MISC': 4, 'B-MISC': 5}[
                single_label]
    return features, num_labels, len(phrase_dict)


def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # Torch 训练优化
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)

    torch.set_default_tensor_type(torch.DoubleTensor)
    x_data, y_data = read_dataset("eng.testa")
    x, y, dict_size = generate_vectors(x_data, y_data)
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    model = LSTMCRF(dict_size, 300, 100, 6).to(device)

    per = 800
    batch_size = 32
    loss_fn = torch.erf
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []
    for epoch in range(100):
        loop = tqdm(total=per // batch_size)
        for i in range(0, per, batch_size):
            this_x, this_y = x[i:i + batch_size], y[i:i + batch_size]

            pred = model(this_x, this_y)
            loss = pred.mean()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            with torch.no_grad():
                result = model(x[:10], None, output_loss=False)
                error = result - y[:10]
                error = torch.sum(torch.abs(error), dim=1)
                loop.set_description(f'Epoch [{epoch}/100]')
                loop.set_postfix(loss=sum(loss_history[-10:]) / len(loss_history[-10:]),
                                 valid_error=torch.sum(error)/10)
                loop.update()
        loop.close()
        torch.save(model.state_dict(), "result.pt")


if __name__ == '__main__':
    main()
