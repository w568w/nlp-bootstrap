import numpy as np
import pandas
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import nn


class CNN(nn.Module):
    def __init__(self, word_num, vector_size, conv2d_output_channel, conv2d_kernel_size, type_num) -> None:
        """

        :param word_num: 每个句子最多有几个单词
        :param vector_size: 每个单词的向量化长度
        :param conv2d_output_channel: conv2d 层的输出通道数
        :param conv2d_kernel_size: conv2d 层的核数
        :param type_num: 分类数
        """
        super().__init__()
        conv2d_output_size = conv2d_output_channel * (word_num - conv2d_kernel_size + 1) * (
                vector_size - conv2d_kernel_size + 1)
        self.net = nn.Sequential(
            nn.Conv2d(1, conv2d_output_channel, conv2d_kernel_size), nn.Flatten(),
            nn.Linear(conv2d_output_size, type_num),
            nn.Softmax()
        )

    def forward(self, x):
        return self.net(x)


def generate_embed_vectors(texts: list[str]) -> ndarray:
    phrase_dict: dict[str, int] = {}
    max_text_len = 0
    for text in texts:
        max_text_len = max(max_text_len, len(text.split()))
        for phrase in text.split():
            if phrase not in phrase_dict:
                phrase_dict[phrase] = len(phrase_dict)
    features = np.zeros((len(texts), 1, max_text_len, len(phrase_dict)))
    for i, text in enumerate(texts):
        for j, phrase in enumerate(text.split()):
            features[i][0][j][phrase_dict[phrase]] = 1
    return features


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    per = 800
    batch_size = 10
    file = pandas.read_csv("train.tsv", sep='\t')[:1000]
    texts = file['Phrase']
    labels = file['Sentiment']
    x = torch.from_numpy(generate_embed_vectors(texts))
    y = torch.from_numpy(labels[:per].values)

    model = CNN(x.shape[2], x.shape[3], 3, 5, 5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    loss_history = []
    for epoch in range(10):
        for i in range(per - batch_size):
            this_x, this_y = x[i:i + batch_size].to(device).type(torch.cuda.FloatTensor), y[i:i + batch_size].to(device)

            pred = model(this_x)
            loss = loss_fn(pred, this_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
        print(loss_history[-1])
    plt.plot(range(len(loss_history)), [sum(loss_history[:i + 1]) / (i + 1) for i in range(len(loss_history))])
    plt.show()


if __name__ == '__main__':
    main()
