import torch
from torch.utils import data
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import constants as cst
import time
from torch.utils import data
from utils.utils_data import one_hot_encoding_type, tanh_encoding_type

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, x, y, seq_size):
        """Initialization""" 
        self.seq_size = seq_size
        self.length = y.shape[0]
        self.x = x
        self.y = y
        if type(self.x) == np.ndarray:
            self.x = torch.from_numpy(x).float()
        if type(self.y) == np.ndarray:
            self.y = torch.from_numpy(y).long()
        self.data = self.x

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, i):
        input = self.x[i:i+self.seq_size, :]
        return input, self.y[i]
    

class GPUBatchLoader:
    """Batches of TLOB windows built on the GPU, a faster replacement for Dataset + DataLoader.

    The whole input tensor is moved to the GPU once; each batch of windows x[i:i+seq_size] is then
    gathered with a single indexing operation instead of one Python call per sample.

    With shuffle=True the batch order is exactly that of DataLoader(shuffle=True, num_workers > 0,
    persistent_workers=True) started from the same global torch RNG state: the first iteration
    draws the DataLoader iterator's base seed, and every epoch draws RandomSampler's seed and uses
    the same torch.randperm. Validation/test (shuffle=False) go in order, as before.

    Multi-GPU (DDP): every rank walks through the same global batches; each takes its slice of
    every training batch, so one step equals one single-GPU step of the full batch. A batch that
    cannot be split evenly over the ranks is processed whole by every rank (same gradient as one
    GPU). Validation and test are not split: every rank sees all of them.
    """

    def __init__(self, x, y, seq_size, batch_size, shuffle, is_train):
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        self.y = y if isinstance(y, torch.Tensor) else torch.from_numpy(y)
        self.x = self.x.float()
        self.y = self.y.long()
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.length = self.y.shape[0]       # number of samples, can be lowered like Dataset.length
        self.device = None
        self.first_iter = True

    def _setup(self):
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        if self.device != device:
            self.x, self.y = self.x.to(device), self.y.to(device)
            self.offsets = torch.arange(self.seq_size, device=device)
            self.device = device
        return rank, world

    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rank, world = self._setup()
        if self.first_iter:
            # DataLoader iterator's base seed (drawn once, persistent workers)
            torch.empty((), dtype=torch.int64).random_()
            self.first_iter = False
        if self.shuffle:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())   # RandomSampler
            generator = torch.Generator()
            generator.manual_seed(seed)
            order = torch.randperm(self.length, generator=generator).to(self.device)
        else:
            order = torch.arange(self.length, device=self.device)
        for start in range(0, self.length, self.batch_size):
            idx = order[start:start + self.batch_size]
            if self.is_train and world > 1 and len(idx) % world == 0:
                idx = idx.tensor_split(world)[rank]
            yield self.x[idx[:, None] + self.offsets], self.y[idx]


class DataModule(pl.LightningDataModule):
    def   __init__(self, train_set, val_set, batch_size, test_batch_size,  is_shuffle_train=True, test_set=None, num_workers=16):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.is_shuffle_train = is_shuffle_train
        if train_set.data.device.type != cst.DEVICE:       #this is true only when we are using a GPU but the data is still on the CPU
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=self.is_shuffle_train,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

        
    