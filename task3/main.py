import collections

import numpy as np
import pandas
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import nn, Tensor
from torchtext import vocab
from tqdm import tqdm


class ESIM(nn.Module):
    def __init__(self, words_dict, word_num, lstm_hidden_size, type_num):
        super().__init__()
        self.word_num = word_num
        vector_size = words_dict.shape[1]
        self.embedding = nn.Embedding.from_pretrained(words_dict)
        self.input_encoding = \
            nn.LSTM(vector_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.composition = \
            nn.LSTM(8 * lstm_hidden_size, lstm_hidden_size, batch_first=True, bidirectional=True)

        self.mlp = nn.Sequential(
            nn.Dropout(),
            nn.BatchNorm1d(lstm_hidden_size * 8),
            nn.Linear(lstm_hidden_size * 8, type_num),
            nn.Tanh(),
            nn.BatchNorm1d(type_num),
            nn.Linear(type_num, type_num),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor):
        # 输入 x (batch_size,2,word_num)
        x = torch.flatten(x, 1, -1)  # (batch_size,2*word_num)
        x = self.embedding(x)  # (batch_size,2*word_num,vector_size)
        x = torch.chunk(x, 2, dim=1)  # A list of (batch_size,word_num,vector_size)
        a_bar, _ = self.input_encoding(x[0])  # (batch_size,word_num,2*lstm_hidden_size)
        b_bar, _ = self.input_encoding(x[1])  # (batch_size,word_num,2*lstm_hidden_size)
        E = torch.matmul(a_bar, torch.transpose(b_bar, 1, 2))  # (batch_size,word_num,word_num)
        a_tilde, b_tilde = torch.matmul(torch.softmax(E, dim=2), b_bar), \
                           torch.matmul(torch.transpose(torch.softmax(E, dim=1), 1, 2),
                                        a_bar)  # (batch_size,word_num,2*lstm_hidden_size)
        m_a, m_b = torch.cat((a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde), dim=2), \
                   torch.cat((b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde),
                             dim=2)  # (batch_size,word_num,8*lstm_hidden_size)
        v_a, _ = self.composition(m_a)  # (batch_size,word_num,lstm_hidden_size*2)
        v_b, _ = self.composition(m_b)  # (batch_size,word_num,lstm_hidden_size*2)

        kernel_size = (self.word_num, 1)
        v = torch.cat((F.avg_pool2d(v_a, kernel_size), F.max_pool2d(v_a, kernel_size), F.avg_pool2d(v_b, kernel_size),
                       F.max_pool2d(v_b, kernel_size)), dim=2)  # (batch_size,1,lstm_hidden_size*8)
        v = torch.squeeze(v, dim=1)  # (batch_size,lstm_hidden_size*8)

        return self.mlp(v)  # (batch_size,type_num)


def clear_str(string: str) -> str:
    return ''.join(filter(str.isalpha, string.lower().strip()))


def generate_vectors(words_dict, jsons) -> tuple[ndarray, list[int]]:
    labels: list[int] = []
    max_text_len = 0
    for jsonObj in jsons:
        s1, s2 = jsonObj['sentence1'], jsonObj['sentence2']
        max_text_len = max(max_text_len, len(s1.split()), len(s2.split()))
        label = jsonObj['gold_label']
        if label == '-':
            label = max(jsonObj['annotator_labels'], key=jsonObj['annotator_labels'].count)
        labels.append({'entailment': 0, 'neutral': 1, 'contradiction': 2}[label])

    features = np.zeros((len(jsons), 2, max_text_len), dtype=np.int64)
    for i, jsonObj in enumerate(jsons):
        strs = [jsonObj['sentence1'], jsonObj['sentence2']]
        for sentence_index in range(2):
            for j, phrase in enumerate(strs[sentence_index].split()):
                try:
                    features[i][sentence_index][j] = words_dict.stoi[clear_str(phrase)]
                except:
                    pass
    return features, labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Torch 训练优化
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)
    data = pandas.read_json("snli_1.0_dev.jsonl", lines=True,
                            typ='series')  # Series 保证 data 的结构是很多行 json，而不是一个表格（Frame）
    # 下载 Glove 变量
    glove = vocab.GloVe(name='6B', dim=300)
    x, y = generate_vectors(glove, data[:2000])
    x = torch.from_numpy(x).to(device)
    y = torch.tensor(y).to(device)
    per = 1600
    batch_size = 32

    model = ESIM(glove.vectors, x.shape[-1], 300, 3).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

    loss_history = []
    for epoch in range(100):
        loop = tqdm(total=per // batch_size)
        for i in range(0, per, batch_size):
            this_x, this_y = x[i:i + batch_size], y[i:i + batch_size]

            pred = model(this_x)
            loss = loss_fn(pred, this_y)
            for param in model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            with torch.no_grad():
                result = torch.argmax(model(x[per:]), dim=-1)
                error = result - y[per:]
                loop.set_description(f'Epoch [{epoch}/100]')
                loop.set_postfix(loss=sum(loss_history[-10:]) / len(loss_history[-10:]),
                                 valid_error=torch.count_nonzero(error) / len(error))
                loop.update()
        loop.close()
        torch.save(model.state_dict(), "result.pt")


if __name__ == '__main__':
    main()
