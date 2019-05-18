"""training models
"""
import pandas as pd
import nibabel as nib
import torch

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def get_arguments():
    return


def get_hyperparameters():
    return


def main():
    """ main interface"""
    # load data files
    bids_annotation = pd.read_table("../classification/results/bids_ready_datasets_annotation.xls")
    print(bids_annotation.shape)
    print(bids_annotation.head())

    image = nib.load(bids_annotation.iloc[:, 8][0])
    print(image.get_data().shape)


main()
