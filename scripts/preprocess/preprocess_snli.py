from genericpath import exists
import os
import json
from transformers import RobertaTokenizer
import torch
import pickle

snli_name = 'snli_1.0_'
labels = {"contradiction": 0, 'neutral': 1, "entailment": 2}
special_tokens = {"start": [0], "pad": [1], "end": [2]}


def preprocess(directory, prefix, tokenizer_type="roberta-large", max_length=512):
    target_file = os.path.join(directory, "{}_{}_{}.pkl".format(
        prefix, tokenizer_type, max_length))
    if exists(target_file):
        return target_file
    with open(os.path.join(directory, snli_name + prefix + '.jsonl'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        pool = []
        for line in lines:
            line_json = json.loads(line)
            if line_json['gold_label'] not in labels:
                continue
            pool.append(
                (line_json['sentence1'], line_json['sentence2'], labels[line_json['gold_label']]))
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_type)
        outpout = []
        for sentence1, sentence2, label in pool:
            if sentence1.endswith("."):
                sentence1 = sentence1[:-1]
            if sentence2.endswith("."):
                sentence2 = sentence2[:-1]
            input1_ids = tokenizer.encode(sentence1, add_special_tokens=False)
            input2_ids = tokenizer.encode(sentence2, add_special_tokens=False)
            input_ids = input1_ids + \
                special_tokens["end"] + input2_ids
            if len(input_ids) > max_length - 2:
                input_ids = input_ids[:max_length - 2]
            length = torch.LongTensor([len(input_ids) + 2])
            input_ids = torch.LongTensor(
                special_tokens["start"] + input_ids + special_tokens["end"])
            label = torch.LongTensor([label])
            outpout.append((input_ids, length, label))
        with open(target_file, "wb") as pkl_file:
            pickle.dump(outpout, pkl_file)
    return target_file
