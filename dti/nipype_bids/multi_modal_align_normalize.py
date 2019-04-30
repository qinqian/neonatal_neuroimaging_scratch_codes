## following process_T1_DTI.py for normalization


import os
import copy
from shutil import copyfile
import numpy as np
import pandas as pd
import nibabel as nb

from nipype.pipeline.engine import Node, MapNode, Workflow
from nipype import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink

from nipype.interfaces.ants import N4BiasFieldCorrection, Registration, DenoiseImage, CreateJacobianDeterminantImage
from nipype.interfaces.fsl import Reorient2Std, ExtractROI, DTIFit, ImageMaths, FLIRT, ApplyWarp, ApplyXFM

from nipype.interfaces.fsl import FAST # accept T1, T2, proton
from nipype.interfaces import fsl
from nipype.pipeline.plugins import MultiProcPlugin
from nipype.utils.filemanip import makedirs
from nipype.workflows.data import get_flirt_schedule
from nipype.workflows.dmri.fsl.utils import enhance
from workflows import ecc_hmc_pipeline, get_container

fsl.FSLCommand.set_default_output_type("NIFTI_GZ")

#from nipype.caching import Memory
#mem = Memory('.')
from nipype import config, logging

config.enable_debug_mode()
input_dir = '/data1/qinqian/MRI/hyperbilirubinemia/output'
output_dir = '/data1/qinqian/MRI/hyperbilirubinemia/output_intra_align'
#output_dir = '/data1/qinqian/MRI/hyperbilirubinemia/output2'

config.set('execution', 'remove_unnecessary_outputs', 'true')
config.set('execution', 'keep_unnecessary_outputs', 'false')
config.set('execution', 'job_finished_timeout', '20')
config.set('execution', 'crashdump_dir', os.path.join(output_dir, 'crash'))
config.set('execution', 'stop_on_first_crash', 'true')
logging.update_logging(config)

def agg_transforms(warp, affine):
    """forward"""
    return [warp, affine]


def inv_agg_transforms(warp, affine):
    """inverse"""
    return [affine, warp]


def main():
    from bids.grabbids import BIDSLayout # not bids validated
    df = pd.read_table('../phaseA_clean_dicom/quality_check.list')
    print(((df.iloc[:,1]==1) & (df.iloc[:, 2] == 1)).sum())
    print(((df.iloc[:,1]==1) | (df.iloc[:, 2] == 1)).sum())
    df = df.loc[(df.iloc[:,1]==1) & (df.iloc[:, 2] == 1), :]
    print(df)

    subjects = df.iloc[:78, 0]
    info = Node(IdentityInterface(fields=['subject_id']), name='subject_id')
    info.iterables = [('subject_id', subjects[:78])]

    templates = {'anat': '{subject_id}/T1/subject_id{subject_id}/*.nii.gz',
                 'b0': '{subject_id}/out_b0/subject_id{subject_id}/*.nii.gz',
                 'FA': '{subject_id}/out_FA/subject_id{subject_id}/*.nii.gz',
                 'MD': '{subject_id}/out_MD/subject_id{subject_id}/*.nii.gz',
                 'mask': '{subject_id}/out_mask/subject_id{subject_id}/*.nii.gz'
                 }

    sf = Node(SelectFiles(templates), name='selectfiles')
    sf.inputs.base_directory = input_dir
    sinker = Node(DataSink(), name='datasink')
    sinker.inputs.base_directory = output_dir
    sinker.inputs.substitutions = [('_subject_id_', 'subject_id')]   # do not need

    preprocess = Workflow(name='preprocess')
    #Normalization to neonate template
    preprocess.base_dir = output_dir

    #intra-subject alignment
    align = Node(FLIRT(), name='intra_align')
    align.inputs.dof = 6

    def align_rest(FA, MD, mask):
        return [FA, MD, mask]

    def get_mask(in_files):
        return in_files[2]

    ## get a list of file names for the FA, MD, mask
    trans = Node(name='align_rest_modality',
                 interface=Function(input_names=['FA', 'MD', 'mask'],
                                    output_names=['trans_modal'],
                                    function=align_rest))
    xfm = MapNode(ApplyXFM(), name='transform',
                  iterfield=['in_file'])
    xfm.inputs.apply_xfm = True

    mult_mask = MapNode(
        fsl.BinaryMaths(operation='mul'),
        name='mask_volumes',
        iterfield=['in_file'])

    mask_T1 = Node()

    # TODO: add prior from age-specific neonatal template to atropos and cortical thickness ,
    #warpTemplate = Node(interface=Registration(), name='warpTemplate', mem_gb=2)
    #warpTemplate.inputs.dimension = 3
    #warpTemplate.inputs.output_transform_prefix = 'ANTS_'

    #warpTemplate.inputs.collapse_output_transforms = False
    #warpTemplate.inputs.transforms =           ['Rigid']
    #warpTemplate.inputs.transform_parameters = [(2.0,)]
    #warpTemplate.inputs.metric = ['Mattes']
    #warpTemplate.inputs.radius_or_number_of_bins = [20]
    #warpTemplate.inputs.metric_weight = [1]
    #warpTemplate.inputs.sampling_strategy = ['Random']
    #warpTemplate.inputs.sampling_percentage = [0.05]
    #warpTemplate.inputs.convergence_threshold = [1.e-8]
    #warpTemplate.inputs.convergence_window_size = [20]

    ### warpTemplate.inputs.initialize_transforms_per_stage = False

    #warpTemplate.inputs.number_of_iterations = [[500, 200]]
    #warpTemplate.inputs.interpolation = 'BSpline'
    #warpTemplate.inputs.float = False

    #warpTemplate.inputs.smoothing_sigmas = [[1,0]]
    #warpTemplate.inputs.sigma_units = ['vox']
    #warpTemplate.inputs.shrink_factors = [[2,1]]
    #warpTemplate.inputs.use_estimate_learning_rate_once = [True]
    #warpTemplate.inputs.use_histogram_matching = [True] # This is the default
    #warpTemplate.inputs.winsorize_upper_quantile = 0.975
    #warpTemplate.inputs.verbose = True # This is the default
    #warpTemplate.config = {'execution': {
    #    'remove_unnecessary_outputs': False
    #   }
    #}
    #      integrate into the nipype pipeline
    #      read through the recon-1, recon-1 nipype pipeline and dhcp-pipeline for wb_command and convert to gii vtk surface
    #      read dhcp-measurement for GI index ...
    # antsAtroposN4.sh -o test -m 3 -n 3 -d 3 -c 3 -x /data1/qinqian/MRI/hyperbilirubinemia/output_intra_align/out_file/subject_id102/_mask_volumes2/sub-102_run-1_dwi_denoised_eccorrect_baseline_brain_mask_flirt_maths.nii.gz -a /data1/qinqian/MRI/hyperbilirubinemia/output/102/T1/subject_id102/sub-102_T1w_noise_corrected_reoriented_enh.nii.gz /data1/qinqian/MRI/hyperbilirubinemia/output_intra_align/out_file/subject_id102/_mask_volumes0/102_FA_flirt_maths.nii.gz /data1/qinqian/MRI/hyperbilirubinemia/output_intra_align/align_b0/subject_id102/sub-102_run-1_dwi_denoised_eccorrect_baseline_flirt.nii.gz
    # KellyKapowski -d 3 -s [testSegmentation.nii.gz,2,1] -g testSegmentationPosteriors2.nii.gz -w testSegmentationPosteriors1.nii.gz -c [30,1e-2,10] -o test_cortical.nii.gz --verbose 1

    sinker = Node(DataSink(), name='datasink')
    sinker.inputs.base_directory = output_dir
    sinker.inputs.substitutions = [('_subject_id_', 'subject_id')]   # do not need


    preprocess.connect([
        (info,           sf,            [('subject_id',          'subject_id')]),
        (sf,             align,         [('anat',                'reference'),
                                         ('b0',                  'in_file')]),
        (align,          sinker,        [('out_file',            'align_b0'),
                                         ('out_matrix_file',     'align_b0.@mat')]),
        (align,          xfm,           [('out_matrix_file',     'in_matrix_file')]),
        (sf,             xfm,           [('anat',                'reference')]),
        (sf,             trans,         [('FA',                  'FA'),
                                         ('MD',                  'MD'),
                                         ('mask',                'mask')]),  # get a list of input names
        (trans,          xfm,           [('trans_modal',         'in_file')]),
        (xfm,            sinker,        [('out_file',            'out_trans')]),

        (xfm,            mult_mask,     [(('out_file', get_mask),'operand_file')]),
        (xfm,            mult_mask,     [('out_file',            'in_file')]),
        (mult_mask,      sinker,        [('out_file',            'out_file')])

        #Normalize to the neonate template
        ##(sf,             warpTemplate,  [('anat',                'fixed_image'),
        ##                                 ('b0',                  'moving_image')]),
        ##(warpTemplate,   sinker,        [('warped_image',        'out_intra')])
    ])

    #initial segmentation multi-modal version
    #atropos

    #normalize to the neonate template

    #warp gray matter density, jacobian determinant images and FA to the template

    #use Nipype internal monitor of the submitted jobs
    #MultiProc always pending too long at the last task
    #because of multiprocessing module issues, expected fix at nipype-2.0
    #reference: https://github.com/nipy/nipype/issues/2548
    #try few n_procs, and mem_gb in Node(), pending too long,
    #preprocess.run('MultiProc', plugin_args={'n_procs': 12,
    #                                         'raise_insufficient':True, 'memory_gb': 50})

    #use SGEGraph to monitor the submitted jobs
    preprocess.run('SGEGraph', plugin_args={'dont_resubmit_completed_jobs': True,
                                            'qsub_args': ' -q normal.q ',
                                            'template':  'template.sh'}) # important PYTHONPATH

    #graph
    #preprocess.write_graph('colored')
    #pre.write_graph(graph2use='flat')

    #dtk.DTITracker
    #MultiProcPlugin

if __name__ == '__main__':
    main()

