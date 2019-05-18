""" loading nifty datasets
"""

import nibabel as nib
from torch.utils import data
import pandas as pd
import numpy as np
import ants

class NiftyLoader(data.Dataset):
    def __init__(self, bids):
        self.bids = pd.read_table(bids)
        self.number = self.bids.shape[0]
        print(self.bids.columns)

    def __getitem__(self, index):
        print(self.bids.iloc[index, 4])
        image = nib.load(self.bids.iloc[index, 4])
        print(index)
        print(image.get_data().shape)
        print(image.get_data().dtype)
        return image.get_data().astype(np.int16)

    def __len__(self):
        return self.number


if __name__ == '__main__':
    nifty = NiftyLoader("../classification/results/bids_ready_datasets_annotation.xls")
    train_loader = data.DataLoader(nifty, shuffle=True, batch_size=16)
    for train_data in train_loader:
        print(train_data.shape)
