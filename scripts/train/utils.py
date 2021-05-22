from typing import List
from tqdm import tqdm
from torch import nn
import torch
import time
import numpy as np
import torch.nn.functional as F


def correct_predictions(probs, labels):
    _, out_classes = probs.max(dim=1)
    correct = (out_classes == labels).sum()
    return correct.item()


def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm,
          regular_lamb):
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
        input_ids, labels, length, start_indexs, end_indexs, span_masks = input_ids.to(device), labels.to(
            device), length.to(device), start_indexs.to(device), end_indexs.to(device), span_masks.to(device)
        y = labels.view(-1)
        y_hat, a_ij = model(input_ids, start_indexs, end_indexs, span_masks)
        # compute loss
        ce_loss = criterion(y_hat, y)
        reg_loss = regular_lamb * a_ij.pow(2).sum(dim=1).mean()
        loss = ce_loss - reg_loss
        # compute acc
        probs = F.softmax(y_hat, dim=1)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, y)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
            input_ids, labels, length, start_indexs, end_indexs, span_masks = input_ids.to(device), labels.to(
                device), length.to(device), start_indexs.to(device), end_indexs.to(device), span_masks.to(device)
            y = labels.view(-1)
            y_hat, a_ij = model(input_ids, start_indexs,
                                end_indexs, span_masks)
            # compute loss
            ce_loss = criterion(y_hat, y)
            loss = ce_loss
            # compute acc
            probs = F.softmax(y_hat, dim=1)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, y)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy


def collate_to_max_length(batch: List[List[torch.Tensor]], max_len: int = None, fill_values: List[float] = None) -> \
        List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    # [batch, num_fields]
    lengths = np.array([[len(field_data)
                         for field_data in sample] for sample in batch])
    batch_size, num_fields = lengths.shape
    fill_values = fill_values or [0.0] * num_fields
    # [num_fields]
    max_lengths = lengths.max(axis=0)
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len

    output = [torch.full([batch_size, max_lengths[field_idx]],
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields)]
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            # seq_length
            data = batch[sample_idx][field_idx]
            output[field_idx][sample_idx][: data.shape[0]] = data
    # generate span_index and span_mask
    max_sentence_length = max_lengths[0]
    start_indexs = []
    end_indexs = []
    for i in range(1, max_sentence_length - 1):
        for j in range(i, max_sentence_length - 1):
            # # span大小为10
            # if j - i > 10:
            #     continue
            start_indexs.append(i)
            end_indexs.append(j)
    # generate span mask
    span_masks = []
    for input_ids, label, length in batch:
        span_mask = []
        middle_index = input_ids.tolist().index(2)
        for start_index, end_index in zip(start_indexs, end_indexs):
            if 1 <= start_index <= length.item() - 2 and 1 <= end_index <= length.item() - 2 and (
                    start_index > middle_index or end_index < middle_index):
                span_mask.append(0)
            else:
                span_mask.append(1e6)
        span_masks.append(span_mask)
    # add to output
    output.append(torch.LongTensor(start_indexs))
    output.append(torch.LongTensor(end_indexs))
    output.append(torch.LongTensor(span_masks))
    # (input_ids, labels, length, start_indexs, end_indexs, span_masks)
    return output


def unit_test():
    input_id_1 = torch.LongTensor([0, 3, 2, 5, 6, 2])
    input_id_2 = torch.LongTensor([0, 3, 2, 4, 2])
    input_id_3 = torch.LongTensor([0, 3, 2])
    batch = [(input_id_1, torch.LongTensor([1]), torch.LongTensor([6])),
             (input_id_2, torch.LongTensor([1]), torch.LongTensor([5])),
             (input_id_3, torch.LongTensor([1]), torch.LongTensor([3]))]

    output = collate_to_max_length(batch=batch, fill_values=[1, 0, 0])
    print(output)
