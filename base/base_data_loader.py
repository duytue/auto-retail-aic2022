import numpy as np
import pytorch_lightning as pl
from box import Box
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(pl.LightningDataModule):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        super().__init__()
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = Box({
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        })

    def setup(self, stage=None):
        if self.validation_split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(self.validation_split, int):
            assert self.validation_split > 0
            assert self.validation_split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = self.validation_split
        else:
            len_valid = int(self.n_samples * self.validation_split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler, valid_sampler = random_split(self.init_kwargs.dataset, [len(train_idx), len(valid_idx)])


        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler

    def train_dataloader(self):
        return DataLoader(self.train_sampler, batch_size=self.init_kwargs.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_sampler, batch_size=self.init_kwargs.batch_size)

    def test_dataloader(self):
        return DataLoader(self.valid_sampler, batch_size=self.init_kwargs.batch_size)

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
