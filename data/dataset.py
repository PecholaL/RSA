""" build Dataset and DataLoader
    * DataLoader provides one audio segment each time
"""

import pickle
import torch
import numpy

from torch.utils.data import Dataset, DataLoader


class CollateFn(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        emb, mel = zip(*batch)
        emb = torch.from_numpy(numpy.array(emb))  # [B, len_emb]
        mel = torch.from_numpy(numpy.array(mel)).transpose(1, 2)  # [B,n_mels,T]
        return emb, mel


class SAdataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        emb, mel = self.data[index]
        return emb, mel

    def __len__(self):
        return len(self.data)


def get_data_loader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=False):
    _collate_fn = CollateFn()
    dataLoader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataLoader


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)
