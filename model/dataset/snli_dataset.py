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

    def __init__(self, prefix: str, directory: str = os.path.join(get_root_path(), "data/dataset/snli_1.0"), tokenizer_type: str = "roberta-base", max_length: int = 64):
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
        sentence1, sentence2, label = self.pool[idx]
        if sentence1.endswith("."):
            sentence1 = sentence1[:-1]
        if sentence2.endswith("."):
            sentence2 = sentence2[:-1]
        input1_dict = self.tokenizer.encode_plus(sentence1,
                                                 # Add '[CLS]' and '[SEP]'
                                                 add_special_tokens=True,
                                                 max_length=self.max_length,
                                                 padding='max_length',
                                                 return_attention_mask=True,
                                                 return_tensors='pt')
        input2_dict = self.tokenizer.encode_plus(sentence2,
                                                 # Add '[CLS]' and '[SEP]'
                                                 add_special_tokens=True,
                                                 max_length=self.max_length,
                                                 padding='max_length',
                                                 return_attention_mask=True,
                                                 return_tensors='pt')

        label = torch.LongTensor([label])
        return input1_dict['input_ids'], input2_dict['input_ids'], input1_dict['attention_mask'], input2_dict['attention_mask'], label
