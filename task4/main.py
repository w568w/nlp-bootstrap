from math import floor
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils import rnn
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, random_split
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


def simple_elementwise_apply(fn: nn.Module, packed_sequence: PackedSequence):
    """applies a point wise function fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)


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
        max_length = torch.zeros((word_len, tag_num))
        last_step = -1 * torch.ones((word_len, tag_num), dtype=torch.int32)
        # 第 0 步
        max_length[0] = self.A[self.n, :self.n] + P[1, :tag_num]
        # 动态规划求下一步
        for t in range(1, word_len):
            for this_index in range(tag_num):
                current_best_length = -10000
                current_best_index = -1
                for last_index in range(tag_num):
                    edge_weight = self.A[last_index][this_index] + P[t + 1][this_index] + max_length[t - 1][last_index]
                    if edge_weight > current_best_length:
                        current_best_index = last_index
                        current_best_length = edge_weight
                max_length[t][this_index] = current_best_length
                last_step[t][this_index] = current_best_index

        # 收束的最后一步
        total_best_length = -10000
        total_best_index = -1
        for last_index in range(tag_num):
            edge_weight = self.A[last_index][self.n + 1] + max_length[-1, last_index]
            if edge_weight > total_best_length:
                total_best_length = edge_weight
                total_best_index = last_index
        path = []
        while total_best_index >= 0:
            path.append(total_best_index)
            total_best_index = last_step[word_len - len(path)][total_best_index]
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
            [torch.tensor(self.viterbi(P[i], real_lengths[i].item(), tag_num_plus_two - 2), dtype=torch.int32) for i in
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
    def __init__(self, dict_size, vector_size, lstm_hidden_size, tag_num):
        super(LSTMCRF, self).__init__()
        self.embedding = nn.Embedding(dict_size, vector_size, padding_idx=-1)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.input_encoding = \
            nn.LSTM(vector_size, lstm_hidden_size, batch_first=True, bidirectional=True, bias=False)
        self.smp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2 * lstm_hidden_size, tag_num, bias=False), nn.Tanh())
        self.crf_layer = CRF(tag_num)

    def forward(self, x: PackedSequence, y: PackedSequence, output_loss=True):
        # 输入 x (batch_size,word_num), y (batch_size,word_num)
        x = simple_elementwise_apply(self.embedding, x)  # (batch_size,word_num,vector_size)
        x, _ = self.input_encoding(x)  # (batch_size,word_num,2 * lstm_hidden_size)
        x = simple_elementwise_apply(self.smp, x)  # (batch_size,word_num,tag_num)
        if y is not None:
            y, y_lens = pad_packed_sequence(y, batch_first=True)
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        return self.crf_layer(y, x,
                              x_lens,
                              output_loss=output_loss)


class CONLLDataset(Dataset):
    def __init__(self, file_path):
        x_data, y_data = self.read_dataset(file_path)
        self.x, self.y, self._dict_size = self.generate_vectors(x_data, y_data)
        # self.x = list(map(lambda x: x.to(device), self.x))
        # self.y = list(map(lambda x: x.to(device), self.y))

    def __getitem__(self, index: Any) -> T_co:
        if isinstance(index, list):
            return [self.x[i] for i in index], [self.y[i] for i in index],
        return self.x[index], self.y[index]

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
                    x_line.append(parts[0].lower())
                    y_line.append(parts[3])
        if len(x_line) > 0 and len(y_line) > 0:
            x_data.append(x_line.copy())
            y_data.append(y_line.copy())
        return x_data, y_data

    def generate_vectors(self, texts: List[List[str]], labels: List[List[str]]) -> Tuple[
        List[Tensor], List[Tensor], int]:
        phrase_dict: Dict[str, int] = {}
        max_text_len = 0
        for i, sentence in enumerate(texts):
            max_text_len = max(max_text_len, len(sentence))
            for phrase in sentence:
                if phrase not in phrase_dict:
                    phrase_dict[phrase] = len(phrase_dict)
        features = []
        for i, sentence in enumerate(texts):
            feature = torch.zeros((len(sentence),), dtype=torch.int32, device="cpu")
            for j, phrase in enumerate(sentence):
                feature[j] = phrase_dict[phrase]
            features.append(feature)
        num_labels = []
        for i, label in enumerate(labels):
            num_label = torch.zeros((len(label),), dtype=torch.int32, device="cpu")
            for j, single_label in enumerate(label):
                num_label[j] = \
                    {'O': 0, 'I-ORG': 1, 'I-PER': 2, 'I-LOC': 3, 'I-MISC': 4, 'B-MISC': 5, 'B-LOC': 6, 'B-ORG': 7,
                     'B-PER': 8}[
                        single_label]
            num_labels.append(num_label)
        return features, num_labels, len(phrase_dict)

    def __len__(self):
        return len(self.x)


def evaluate_confusion(model: nn.Module, tag_num: int, x: PackedSequence, y: PackedSequence):
    cf = torch.zeros((tag_num, tag_num))
    result = model(x, None, output_loss=False).int()
    y, y_lens = pad_packed_sequence(y, batch_first=True)
    x, x_lens = pad_packed_sequence(x, batch_first=True)
    batch_size = x.shape[0]
    for i in range(batch_size):
        for j in range(x_lens[i].item()):
            cf[y[i][j]][result[i][j]] += 1
    return cf


def evaluate_f1(confusion_matrix: Tensor):
    tag_num = confusion_matrix.shape[0]
    f1 = []
    for tag in range(tag_num):
        predict_num = torch.sum(confusion_matrix[:, tag]).item()
        actual_num = torch.sum(confusion_matrix[tag]).item()
        correct_num = confusion_matrix[tag][tag].item()
        try:
            precision = predict_num / correct_num
            recall = actual_num / correct_num
            f1.append(2.0 / (precision + recall))
        except ZeroDivisionError:
            f1.append(-1)
    return f1


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
    return pack_sequence_cpu(x).to(device), pack_sequence_cpu(y).to(device)


def main():
    # Torch 训练优化
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)

    if device == "cuda":
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
    dataset = CONLLDataset("eng.train")
    per = floor(0.8 * len(dataset))
    train_dataset, valid_dataset = random_split(dataset, (per, len(dataset) - per), generator=torch.Generator(device))
    model = LSTMCRF(dataset.dict_size(), 300, 100, 9).to(device)

    batch_size = 8
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=pack_collate_fn)
    loss_history = []
    for epoch in range(100):
        loop = tqdm(total=len(train_loader))
        for i, data in enumerate(train_loader):
            loss = model(*data)
            # pred = evaluate(model, *data)
            # 增加惩罚项，迫使网络不要输出 0
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            loss_history.append(loss.item())
            loop.set_description(f'Epoch [{epoch}/100]')
            loop.set_postfix(loss=sum(loss_history[-10:]) / len(loss_history[-10:]))
            loop.update()
        print(evaluate_f1(evaluate_confusion(model, 9, *pack_two(*valid_dataset[:50]))))
        loop.close()
        torch.save(model.state_dict(), "result.pt")


if __name__ == '__main__':
    main()
