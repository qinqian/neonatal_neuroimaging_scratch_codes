from argparse import ArgumentParser
from typing import Dict, Any

import ants
import subprocess
import os

import argparse
import pandas as pd
from pathlib import Path

from skimage.restoration import denoise_wavelet
import numpy as np
from skimage import exposure


def enhan_ants_image(ants_image, mask, clip_limit=0.015):
    im = ants_image.denoise_image()

    imdata = im.numpy()
    imshape = im.shape

    immin = imdata.min()
    #imdata = (imdata - immin).astype(np.uint16)
    imdata = (imdata - immin)

    adapted = exposure.equalize_adapthist(
        denoise_wavelet(imdata.reshape(imshape[0], -1)), clip_limit=clip_limit)

    adapted_image = adapted.reshape(imshape)
    return ants.make_image(mask.shape, voxval=adapted_image,
                           spacing=(1, 1, 1),
                           direction=ants.get_direction(mask),
                           origin=ants.get_origin(mask))


def load_iso_resample(image):
    return ants.resample_image(ants.image_read(image), (1, 1, 1), use_voxels=False)


def return_path(modality, output_dir, output_id):
    output_file = os.path.join(output_dir, f"{output_id}_{modality}_processed.nii.gz")
    return output_file


def write_images(image, modality, output_dir, output_id):
    ants.image_write(image, return_path(modality, output_dir, output_id))


class SubjectScan(object):
    def __init__(self, input_id: int, template_id: int, output_dir: str, bids_table: pd.DataFrame):
        self.input_id = input_id
        self.output_dir = output_dir
        self.template = template_id
        self.bids = bids_table

    @property
    def norm_to_template(self):

        # use exist template
        if all([os.path.exists(return_path(modality, self.output_dir, self.template))
                for modality in ['t1', 't2', 't1s', 't2f', 'adc']]):
            template = {'t1':  ants.image_read(return_path('t1', self.output_dir, self.template)),
                        't2':  ants.image_read(return_path('t2', self.output_dir, self.template)),
                        't1s': ants.image_read(return_path('t1s', self.output_dir, self.template)),
                        't2f': ants.image_read(return_path('t2f', self.output_dir, self.template)),
                        'adc': ants.image_read(return_path('adc', self.output_dir, self.template))}

        else:
            # preprocessed template images
            template = self.intrasubject_align(self.template)
            for modality in ['t1', 't2', 't1s', 't2f', 'adc']:
                write_images(template[modality], modality, self.output_dir, self.template)

        in_image = self.intrasubject_align(self.input_id)
        standard_dict = {}
        reg = ants.registration(template['t2'], in_image['t2'],
                                type_of_transform='Affine',
                                outprefix=str(self.input_id))
        standard_dict['t2'] = reg['warpedmovout']

        for modality in ['t1', 't1s', 't2f', 'adc']:
            standard_dict[modality] = ants.apply_transforms(template['t2'],
                                                            in_image[modality],
                                                            reg['fwdtransforms'])
        for modality in ['t1', 't2', 't1s', 't2f', 'adc']:
            write_images(standard_dict[modality], modality, self.output_dir, self.input_id)
        return standard_dict

    def intrasubject_align(self, image_id) -> dict:
        t2_file = self.bids.loc[image_id, ].loc['t2']
        t2_prefix = t2_file.replace('.nii.gz', '_processed')

        subprocess.call(f'bet {t2_file} {t2_prefix} -m -R -f 0.4', shell=True)
        image_modality_dict: Dict[str, Any] = {'t2': load_iso_resample(t2_file),
                                               'mask': load_iso_resample(f'{t2_prefix}_mask.nii.gz')}

        ants.image_write(image_modality_dict['t2'], f'{t2_prefix}_initial.nii.gz')
        for modality in ['t1', 't1s', 't2f', 'adc']:
            if modality in ['adc', 't1s']:
                # ADC sometimes failed rigid registration with ANTS
                # use flirt instead
                image_file = self.bids.loc[image_id, ].loc[modality]
                image_prefix = image_file.replace('.nii.gz', '_processed')

                subprocess.call(f"flirt -datatype float -interp spline -coarsesearch 90 -finesearch 90 -dof 6 -in "
                                f"{image_file} "
                                f"-ref {t2_prefix}_initial.nii.gz -out {image_prefix}.nii.gz", shell=True)
                image_modality_dict[modality] = ants.image_read(f"{image_prefix}.nii.gz")

            else:
                image_modality_dict[modality] = ants.image_read(self.bids.loc[image_id,].loc[modality])
                image_modality_dict[modality] = ants.registration(image_modality_dict['t2'],
                                                                  image_modality_dict[modality],
                                                                  type_of_transform='Rigid', outprefix=str(image_id))
                image_modality_dict[modality] = image_modality_dict[modality]['warpedmovout']

        for modality in ['t2', 't1', 't1s', 't2f', 'adc', 'mask']:
            image_modality_dict[modality] = image_modality_dict[modality] * image_modality_dict['mask']
            image_modality_dict[modality] = image_modality_dict[modality].crop_image(image_modality_dict['mask'])

        for modality in ['t2', 't1', 't1s', 't2f', 'adc']:
            image_modality_dict[modality] = enhan_ants_image(image_modality_dict[modality], image_modality_dict['mask'])
        return image_modality_dict


def get_parameters():
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--bids', required=True, type=str)
    parser.add_argument('--inputid', required=True, type=int)
    parser.add_argument('--templateid', default=635, type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_parameters()
    bids_table = pd.read_csv(args.bids, index_col=0)
    candidate_image_dict = SubjectScan(args.inputid, template_id=args.templateid,
                                       output_dir=os.path.abspath('results/processed'), bids_table=bids_table)
    candidate_image_dict.norm_to_template
    return 0


if __name__ == '__main__':
    main()
