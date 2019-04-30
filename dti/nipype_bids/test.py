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
from nipype.interfaces.fsl import Reorient2Std, ExtractROI, DTIFit, ImageMaths

from nipype.interfaces.fsl import FAST # accept T1, T2, proton
from nipype.interfaces import fsl
from nipype.pipeline.plugins import MultiProcPlugin
from nipype.utils.filemanip import makedirs
from nipype.workflows.dmri.fsl.utils import enhance

from mrtrix3 import dwidenoise
from workflows import ecc_hmc_pipeline, get_container

fsl.FSLCommand.set_default_output_type("NIFTI_GZ")

#from nipype.caching import Memory
#mem = Memory('.')
from nipype import config, logging

#config.enable_debug_mode()
input_dir = '/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/'
#output_dir = '/data1/qinqian/MRI/hyperbilirubinemia/output'
output_dir = '/data1/qinqian/MRI/hyperbilirubinemia/test'

config.set('execution', 'remove_unnecessary_outputs', 'true')
config.set('execution', 'keep_unnecessary_outputs', 'false')
config.set('execution', 'job_finished_timeout', '20')
config.set('execution', 'crashdump_dir', os.path.join(output_dir, 'crash'))
config.set('execution', 'stop_on_first_crash', 'true')
logging.update_logging(config)

conf = {
    'template_t1': '/home/qinqian/data1/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/NeonatalAtlas2/template_T1.nii.gz'
}

def agg_transforms(warp, affine):
    """forward"""
    return [warp, affine]


def inv_agg_transforms(warp, affine):
    """inverse"""
    return [affine, warp]


def main():
    from bids.grabbids import BIDSLayout # not bids validated
    layout = BIDSLayout(input_dir)
    print(layout.get_subjects())
    print(layout.get_modalities())
    df = pd.read_table('../phaseA_clean_dicom/quality_check.list')
    print(((df.iloc[:,1]==1) & (df.iloc[:, 2] == 1)).sum())
    print(((df.iloc[:,1]==1) | (df.iloc[:, 2] == 1)).sum())

    df = df.loc[(df.iloc[:,1]==1) & (df.iloc[:, 2] == 1), :]
    print(df)

    subjects = df.iloc[:78, 0]
    #print(subjects)
    #subjects = subjects[~np.isin(subjects, np.array([332]))]
    #print(len(subjects))

    info = Node(IdentityInterface(fields=['subject_id']), name='subject_id')
    info.iterables = [('subject_id', subjects[:2])]

    ## test 332 for hmc N4BiasFieldCorrection
    #info.iterables = [('subject_id', ['332', '322'])]

    templates = {'anat': 'sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz',
                 'dwi': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dwi.nii.gz',
                 'bvals': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dwi.bval',
                 'bvecs': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dwi.bvec'}

    sf = Node(SelectFiles(templates), name='selectfiles')
    sf.inputs.base_directory = input_dir
    sinker = Node(DataSink(), name='datasink')
    sinker.inputs.base_directory = output_dir
    sinker.inputs.substitutions = [('_subject_id_', 'subject_id')]   # do not need

    preprocess = Workflow(name='preprocess')
    #Normalization to neonate template
    preprocess.base_dir = output_dir

    denoise_anat = Node(DenoiseImage(), name='denoise_anat', mem_gb=2)
    denoise_anat.inputs.dimension = 3

    ## anat specific
    reorient = Node(Reorient2Std(), name="reorient")
    enhan = Node(Function(function=enhance,
                          input_names=['in_file'],
                          output_names=['out_file']), name="enhance")
    enhan.inputs.clip_limit = 0.015

    #intra-subject alignment
    #warpTemplate = Node(interface=Registration(), name='warpTemplate')
    #warpTemplate.inputs.dimension = 3
    #warpTemplate.inputs.metric = ['CC', ]
    #warpTemplate.inputs.metric_weight = [1.0, ]
    #warpTemplate.config = {'execution': {
    #    'remove_unnecessary_outputs': False
    #}}

    #initial segmentation multi-modal version
    # atropos

    #normalize to the neonate template

    #warp gray matter density, jacobian determinant images and FA to the template


    #preprocess dwis
    dwi_workflow = ecc_hmc_pipeline(name='dwi_workflow')
    #dwi_workflow.base_dir = output_dir

    denoise = Node(dwidenoise(), name='denoise')
    #test denoise
    denoised = True
    if denoised:
        preprocess.connect([
            (info, sf, [('subject_id', 'subject_id')]),
            ##test denoise
            (sf, denoise,            [('dwi', 'in_file')]),
            (denoise, sinker,        [('out_file', 'denoised_output')]),
        ])

    #use Nipype internal monitor of the submitted jobs
    #MultiProc always pending too long at the last task
    #because of multiprocessing module issues, expected fix at nipype-2.0
    #reference: https://github.com/nipy/nipype/issues/2548
    #try few n_procs, and mem_gb in Node() and memory_gb in run
    #preprocess.run('MultiProc', plugin_args={'n_procs': 16, 'raise_insufficient':True, 'memory_gb': 48})

    # plugin IPython error, ipyparallel.error.RemoteError: IndexError(msg pending: 'f3fc415e-9f826b74f1881a4c278e971b')
    # cannot fix yet
    # preprocess.run('IPython', plugin_args={'n_procs': 16})

    #SGE sometimes crash for waiting too many queues, don't know why yet
    #error /data1/qinqian/MRI/hyperbilirubinemia/output/preprocess/dwi_workflow/_subject_id_322/denoise missing directory
    #without customize denoise step, this can succeed...
    preprocess.run('SGE', plugin_args={'qsub_args': ' -cwd -q normal.q ',
                                       'template':  'template.sh',
                                       'overwrite': True})

    #use SGEGraph to monitor the submitted jobs
    #preprocess.run('SGEGraph', plugin_args={'dont_resubmit_completed_jobs': True, 'qsub_args': ' -q normal.q '})

if __name__ == '__main__':
    main()


