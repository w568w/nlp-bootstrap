import math
import os.path
from typing import Union, Dict, Tuple, Optional, Sequence, List, Any, Set

import numpy as np
import pandas
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pandas import Series
from pytorch_lightning.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS
from sklearn.metrics import f1_score, precision_score, \
    recall_score
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

CKPT_PATH = "newest.ckpt"


def f1_loss(y_true: Tensor, y_pred: Tensor, eps: float = 1e-7) -> Tensor:
    """
    F1-like loss function.
    :param y_true: (batch_size,class_num)
    :param y_pred: (batch_size,class_num)
    :param eps: small correction for zero.
    :return:
    """
    tp = torch.sum(y_true * y_pred, 0)
    fp = torch.sum((1 - y_true) * y_pred, 0)
    fn = torch.sum(y_true * (1 - y_pred), 0)

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)


class Classifier(pl.LightningModule):

    def __init__(self, bert: BertModel, class_num: int, pos_weight: Optional[Tensor], learning_rate: float = 1e-2,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.lr = learning_rate
        self.bert = bert
        self.smp = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, class_num)
        )

    def generate_padding_mask(self, real_lengths: Tensor, max_len: int) -> Tensor:
        real_lengths = real_lengths.cpu()
        batch_size = real_lengths.shape[0]
        mask = torch.zeros((batch_size, max_len), device=self.device, dtype=torch.int)
        for i in range(batch_size):
            mask[i, :real_lengths[i]] = 1
        return mask

    def forward(self, x: PackedSequence) -> Any:
        x, x_lens = pad_packed_sequence(x, batch_first=True)  # (batch_size,max_len),(batch_size)
        bert_output: BaseModelOutputWithPoolingAndCrossAttentions = \
            self.bert(input_ids=x, attention_mask=self.generate_padding_mask(x_lens,
                                                                             torch.max(
                                                                                 x_lens).int()))  # (batch_size,hidden_size)
        return self.smp(bert_output.pooler_output)  # (batch_size,class_num)

    def training_step(self, batch: Tuple[PackedSequence, Tensor], batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        x, y = batch
        output = self.forward(x)  # (batch_size,class_num)
        loss = F.binary_cross_entropy_with_logits(output, y.float(), pos_weight=self.pos_weight.to(self.device))
        return loss

    def validation_step(self, batch: Tuple[PackedSequence, Tensor], batch_id, *args, **kwargs):
        x, y = batch
        output = self.forward(x)  # (batch_size,class_num)
        metrics = multi_label_metrics(output, y)
        self.log_dict(metrics, prog_bar=True, batch_size=output.shape[0])

    def configure_optimizers(self) -> Optional[Union[
        Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]
    ]]:
        return torch.optim.Adam(self.parameters(), self.lr)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.trainer.save_checkpoint(CKPT_PATH)


class HoleDataset(Dataset):
    @staticmethod
    def create_or_load(file_path: str, tokenizer: PreTrainedTokenizer, save_path: Optional[str]) -> "HoleDataset":
        if save_path and os.path.exists(save_path):
            return torch.load(save_path)
        else:
            dataset = HoleDataset(file_path, tokenizer)
            torch.save(dataset, save_path)
            return dataset

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        json_array: Series = pandas.read_json(file_path, typ="series")
        self.idx2tag, self.tag2idx, self.tag_num = self.collect_tags(json_array)
        self.tag_weight = self.generate_tag_weight(json_array)
        self.x, self.y = self.generate_vectors(json_array, tokenizer)

    def generate_tag_weight(self, json_array: Series, clipping: float = 10.0):
        tags_freq = [0] * self.tag_num
        tag_weight = torch.zeros(self.tag_num, dtype=torch.float)
        for item in json_array:
            for tag in item["label"]:
                tags_freq[self.tag2idx[tag]] += 1
        tag_freq_max = max(tags_freq)
        for i in range(self.tag_num):
            tag_weight[i] = min(tag_freq_max / tags_freq[i], clipping)
        return tag_weight

    def generate_vectors(self, json_array: Series, tokenizer: PreTrainedTokenizer):
        features = []
        tags = []
        for item in json_array:
            features.append(torch.squeeze(tokenizer(item["text"], return_tensors='pt').input_ids, dim=0)[:512])
            tag_vec = torch.zeros(self.tag_num, dtype=torch.float)
            for tag in item["label"]:
                tag_vec[self.tag2idx[tag]] = 1
            tags.append(tag_vec)
        return features, tags

    @staticmethod
    def collect_tags(json_array: Series) -> Tuple[Dict, Dict, int]:
        tag_set: Set[str] = set()
        for item in json_array:
            tag_set.update(item["label"])
        idx2tag = dict((i, tag) for i, tag in enumerate(tag_set))
        tag2idx = dict((tag, i) for i, tag in enumerate(tag_set))
        return idx2tag, tag2idx, len(tag_set)

    def __getitem__(self, index: Any) -> Tuple[List[Tensor], List[Tensor]]:
        if isinstance(index, list):
            return [self.x[i] for i in index], [self.y[i] for i in index],
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class LitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dataset, train_val_ratio: float = 0.8):
        super().__init__()
        total_num = len(train_dataset)
        train_num = math.floor(total_num * train_val_ratio)
        self.train_dataset, self.val_dataset = random_split(train_dataset, lengths=[train_num, total_num - train_num])
        self.batch_size = batch_size

    def train_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pack_collate_fn,
                          pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pack_collate_fn,
                          pin_memory=True)


class PredictionCallback(pl.Callback):
    def __init__(self, dataset: HoleDataset, tokenizer: BertTokenizer, predict_input: str,
                 predict_every_n_batches: int = 50, ):
        self.predict_input = torch.squeeze(tokenizer(predict_input, return_tensors='pt').input_ids, dim=0)[:512]
        self.predict_input = pack_sequence_cpu([self.predict_input])
        self.dataset = dataset
        self.predict_every_n_batches = predict_every_n_batches

    def predict(self, model: "pl.LightningModule"):
        output = torch.squeeze(model(self.predict_input.to(model.device)), dim=0)
        values, indices = torch.topk(output, 10)
        tags = [self.dataset.idx2tag[i.item()] for i in indices]

        print(list(zip(tags, values.tolist())))

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                           batch: Any, batch_idx: int, unused: int = 0) -> None:
        if batch_idx % self.predict_every_n_batches == 0:
            self.predict(pl_module)
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


def pack_sequence_cpu(sequences, enforce_sorted=True):
    lengths = torch.as_tensor([v.size(0) for v in sequences], device=torch.device("cpu"))
    return pack_padded_sequence(pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted)


def pack_collate_fn(data):
    return pack_two(*zip(*data))


def pack_two(x, y):
    xy = list(zip(x, y))
    xy.sort(key=lambda z: len(z[0]), reverse=True)
    x, y = zip(*xy)
    return pack_sequence_cpu(x), torch.stack(y, dim=0)


def multi_label_metrics(predictions, labels, threshold=0.3):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = torch.sigmoid(predictions).cpu().numpy()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros_like(probs)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels.cpu().numpy()
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    # return as dictionary
    metrics = {'f1': f1,
               'precision': precision, 'recall': recall}
    return metrics


EPOCHS = 10


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    dataset = HoleDataset.create_or_load("data/labels.json", tokenizer, "dataset.pt")
    model = Classifier(bert_model, class_num=dataset.tag_num, learning_rate=2e-5, pos_weight=dataset.tag_weight)
    datamodule = LitDataModule(batch_size=32, train_dataset=dataset, train_val_ratio=0.9)
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="gpu", devices=1,
                         callbacks=[PredictionCallback(dataset, tokenizer, "一句测试文本")])
    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)

    def predict(text: str):
        text = torch.squeeze(tokenizer(text, return_tensors='pt').input_ids, dim=0)[:512]
        text = pack_sequence_cpu([text])
        output = torch.squeeze(model(text), dim=0)
        values, indices = torch.topk(output, 10)
        tags = [dataset.idx2tag[i.item()] for i in indices]
        print(list(zip(tags, values.tolist())))


if __name__ == '__main__':
    main()
