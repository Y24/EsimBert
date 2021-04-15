import torch
from typing import List, Tuple
import random


def pack_data(batch: List[Tuple[torch.Tensor]], fill_value=1):
    batch_size = len(batch)
    for i in range(batch_size):
        output[i][:lengths[i]] = batch[i][0]
    return [(output[i], batch[i][1], batch[i][2]) for i in range(batch_size)]


if __name__ == "__main__":
    size = [random.randint(3, 6) for _ in range(4)]
    batch_size = random.randint(3, 6)
    batch_demo = [torch.randint(3, 7, [i]) for i in size]
    output, lengths = pack_data(batch_demo)
