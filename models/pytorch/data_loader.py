""" Loading nifty datasets 
https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
"""
import ants
from torch.utils import data
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform 
from batchgenerators.transforms.noise_transforms import RicianNoiseTransform
from batchgenerators.transforms.resample_transforms import ResampleTransform
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
np.random.seed(1)

def get_transforms():
    all_transforms = Compose([
      #RndTransform(MirrorTransform(axes=(0,1)), prob=0.5),
      #RndTransform(ContrastAugmentationTransform((0.3, 3.), preserve_range=True), prob=0.5),
      #RndTransform(RicianNoiseTransform(noise_variance=(0, 200))), 
      #RndTransform(ResampleTransform(zoom_range=(0.05, 0.2)))
      MirrorTransform(axes=(0,1)),
      BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15),
      GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15),
      GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15),
      GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15),
      GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5, p_per_sample=0.15)
    ])
    return all_transforms

IMAGES_MEAN_SD = {0: [0.5872402, 0.2024262],
                  1: [0.5821896, 0.1871165],
                  2: [0.4300238, 0.2120626],
                  3: [0.4056657, 0.1636801],
                  4: [0.3866455, 0.133584]}

MACHINE_INDEX  = {'SIEMENS-Avanto-1.5T': 0,
                  'GE MEDICAL SYSTEMS-DISCOVERY MR750-3.0T': 1,
                  'SIEMENS-Skyra-3.0T': 2}


class NiftyLoaderTransform(DataLoader):
    def __init__(self, bids, modal, symptom, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1, return_incomplete=False, shuffle=True, infinite=True):
        self.nifty = NiftyLoader(bids, modal, symptom, 635) # fix template id
        super().__init__(self.nifty.bids.index, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle, infinite)
        self.indices = list(range(len(self.nifty)))
        self.num_modal = len(modal)

    def generate_train_batch(self):
        idx = self.get_indices()
        imgs = np.zeros((self.batch_size, self.num_modal, *self.nifty.mask.shape))
        labels = np.zeros((self.batch_size, 2))

        for i, index in enumerate(idx):
            image, label = self.nifty[index]
            imgs[i] = image
            labels[i] = label
        # now construct the dictionary and return it. np.float32 cast because most networks take float
        return {'data': imgs.astype(np.float32), 'labels': labels.astype(np.float32)}


class NiftyLoader(data.Dataset):
    """ class for loading nifty """
    def __init__(self, bids, modalities, symptom, template_id):
        """ bids: pd.DataFrame
        modalities: a list of int for modalities index
        symptom:    column name for symptom in bids table
        """
        self.bids = bids
        self.number = self.bids.shape[0]
        self.modalities = sorted(modalities)
        self.symptom = symptom
        self.mask = ants.image_read(f'results/processed/{template_id}_t2_processed.nii.gz')
        self.mask = self.mask.get_mask().numpy()

    def __getitem__(self, index):
        row_index = self.bids.index[index]
        # machine = machine_index[self.bids.iloc[index, ].loc['merge_machine_info']]
        all_modalities = [f"results/processed/{row_index}_t2f_processed.nii.gz", # FLAIR
                          f"results/processed/{row_index}_t2_processed.nii.gz",  # T2
                          f"results/processed/{row_index}_adc_processed.nii.gz", # ADC
                          f"results/processed/{row_index}_t1_processed.nii.gz",  # T1
                          f"results/processed/{row_index}_t1s_processed.nii.gz"]  # T1s
        all_modalities = [all_modalities[m] for m in self.modalities]
        image = [((ants.image_read(m).numpy()-IMAGES_MEAN_SD[stat][0])/IMAGES_MEAN_SD[stat][1])  ### *self.mask, ants mask is weird sometimes, 
                 for m, stat in zip(all_modalities, self.modalities)]
        image = np.stack(image, axis=0)
        label = self.bids.iloc[index, :].loc[self.symptom].astype('float32')
        label = np.array([1-label, label], dtype='float32')
        return [image, label]

    def __len__(self):
        return self.number


if __name__ == '__main__':
    import pandas as pd
    ALL_BIDS = pd.read_csv("results/bids_head1.csv", index_col=0)
    EXCLUDE_ID = "check_datasets/bad_ids_thrown_ids_for_antspy2.txt"
    with open(EXCLUDE_ID) as fin:
        EXCLUDE_IDS = list(map(int, fin.readlines()))
    ALL_BIDS = ALL_BIDS.loc[~ALL_BIDS.index.isin(EXCLUDE_IDS), :]

    NIFTY = NiftyLoader(ALL_BIDS,
                        modalities=[0], symptom="Leukomalacia")
    train_loader = data.DataLoader(NIFTY, shuffle=True, batch_size=8, num_workers=8)
    for train_data, labels in train_loader:
        print(train_data.shape, labels.shape)
        break
