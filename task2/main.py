import collections
from typing import Tuple

import numpy as np
import pandas
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import nn


class CNN(nn.Module):
    def __init__(self, dict_size, word_num, vector_size, conv2d_output_channel, conv2d_kernel_sizes, type_num) -> None:
        """

        :param word_num: 每个句子最多有几个单词
        :param vector_size: 每个单词的向量化长度
        :param conv2d_output_channel: conv2d 层的输出通道数
        :param conv2d_kernel_sizes: conv2d 层的核数
        :param type_num: 分类数
        """
        super().__init__()
        self.embedding = nn.Embedding(dict_size, vector_size, padding_idx=0)
        self.cnn_net = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, conv2d_output_channel, (kernel, vector_size)),
            nn.ReLU(),  # (batch_size,conv2d_output_channel,word_num-kernel_size+1,1)
            nn.Flatten(start_dim=2),  # (batch_size,conv2d_output_channel,word_num-kernel+1)
            nn.MaxPool2d((1, word_num - kernel + 1)),  # (batch_size,conv2d_output_channel,1)
            nn.Flatten(start_dim=1)  # (batch_size,conv2d_output_channel)
        ) for kernel in conv2d_kernel_sizes])
        self.linear_output = nn.Sequential(
            nn.Dropout(),
            nn.Linear(conv2d_output_channel * len(conv2d_kernel_sizes), type_num),  # (batch_size,type_num)
            nn.Softmax()
        )

    def forward(self, x):
        # 输入 x(batch_size,word_num)
        x = self.embedding(x)  # (batch_size,word_num,vector_size)
        x = torch.unsqueeze(x, 1)  # (batch_size,1,word_num,vector_size)
        x = [cnn(x) for cnn in
             self.cnn_net]  # a list of (batch_size,conv2d_output_channel), length = len(conv2d_kernel_sizes)
        x = torch.cat(x, dim=1)  # (batch_size,conv2d_output_channel*len(conv2d_kernel_sizes))
        return self.linear_output(x)


def generate_vectors(texts: list[str]) -> tuple[ndarray, int]:
    phrase_dict: dict[str, int] = {}
    max_text_len = 0
    for text in texts:
        max_text_len = max(max_text_len, len(text.split()))
        for phrase in text.split():
            if phrase not in phrase_dict:
                phrase_dict[phrase] = len(phrase_dict)
    features = np.zeros((len(texts), max_text_len), dtype=np.int64)
    for i, text in enumerate(texts):
        for j, phrase in enumerate(text.split()):
            features[i][j] = phrase_dict[phrase]
    return features, len(phrase_dict)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    per = 800
    batch_size = 50
    file = pandas.read_csv("train.tsv", sep='\t')[:1000]
    texts = file['Phrase']
    labels = file['Sentiment']
    vec, dict_num = generate_vectors(texts)
    x = torch.from_numpy(vec).to(device)
    y = torch.from_numpy(labels[:per].values).to(device)

    model = CNN(dict_num, x.shape[-1], 128, 2, [2, 3, 4], 5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    loss_history = []
    for epoch in range(100):
        for i in range(per - batch_size):
            this_x, this_y = x[i:i + batch_size], y[i:i + batch_size]

            pred = model(this_x)
            loss = loss_fn(pred, this_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
        result = torch.argmax(model(x[per:]), dim=-1).cpu()
        print(collections.Counter(result.numpy()))
        print(collections.Counter(labels[per:]))
        error = result - labels[per:]
        print("Error: %d/%d" % (np.count_nonzero(error), len(error)))
    plt.plot(range(len(loss_history)), [sum(loss_history[:i + 1]) / (i + 1) for i in range(len(loss_history))])
    # plt.plot(range(len(loss_history)), loss_history)
    plt.show()


if __name__ == '__main__':
    main()
