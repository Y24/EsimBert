import json
import os
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from util.path import get_root_path


class SNLIDataset(Dataset):
    snli_name = 'snli_1.0_'
    labels = {"contradiction": 0, 'neutral': 1, "entailment": 2}
    special_tokens = {"start": [0], "pad": [1], "end": [2]}

    def __init__(self, prefix: str, directory: str = os.path.join(get_root_path(), "data/dataset/snli_1.0"), tokenizer_type: str = "roberta-base", max_length: int = 512):
        super().__init__()
        print("Init SNLIDataset...")

        self.max_length = max_length
        with open(os.path.join(directory, self.snli_name + prefix + '.jsonl'), 'r', encoding='utf8') as f:
            lines = f.readlines()
        self.pool = []
        for line in lines:
            line_json = json.loads(line)
            if line_json['gold_label'] not in self.labels:
                continue
            self.pool.append(
                (line_json['sentence1'], line_json['sentence2'], self.labels[line_json['gold_label']]))
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_type)
        print("SNLIDataset inited!")

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, idx: int):
        premise, hypothesis, label = self.pool[idx]
        if premise.endswith("."):
            premise = premise[:-1]
        if hypothesis.endswith("."):
            hypothesis = hypothesis[:-1]
        premise_ids = self.tokenizer.encode(
            premise, add_special_tokens=False)
        hypothesis_ids = self.tokenizer.encode(
            hypothesis, add_special_tokens=False)
        input_ids = premise_ids + [2] + hypothesis_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([label])
        return input_ids, label, length