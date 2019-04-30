import json
import os
import copy
from shutil import copyfile
import numpy as np
import pandas as pd
import nibabel as nb

from nipype.pipeline.engine import Node, MapNode, Workflow
from nipype import IdentityInterface, Function, Select
from nipype.interfaces.io import SelectFiles, DataSink

from nipype.interfaces.ants import N4BiasFieldCorrection, Registration, DenoiseImage, CreateJacobianDeterminantImage, \
    ApplyTransforms
from nipype.interfaces.fsl import Reorient2Std, ExtractROI, DTIFit, ImageMaths

from nipype.interfaces.fsl import FAST # accept T1, T2, proton
from nipype.interfaces import fsl
from nipype.pipeline.plugins import MultiProcPlugin
from nipype.utils.filemanip import makedirs
from nipype.workflows.dmri.fsl.utils import enhance
from workflows import ecc_hmc_pipeline, get_container

fsl.FSLCommand.set_default_output_type("NIFTI_GZ")

#from nipype.caching import Memory
#mem = Memory('.')
from nipype import config, logging

config.enable_debug_mode()
input_dir = '/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/'
output_dir = '/data1/qinqian/MRI/hyperbilirubinemia/output'
#output_dir = '/data1/qinqian/MRI/hyperbilirubinemia/output2'

config.set('execution', 'remove_unnecessary_outputs', 'true')
config.set('execution', 'keep_unnecessary_outputs', 'false')
config.set('execution', 'job_finished_timeout', '20')
config.set('execution', 'crashdump_dir', os.path.join(output_dir, 'crash'))
config.set('execution', 'stop_on_first_crash', 'true')
logging.update_logging(config)

with open('template.json') as j:
    conf = json.loads(j.read())
print(conf)

def agg_transforms(warp, affine):
    """forward"""
    return [warp, affine]


def inv_agg_transforms(warp, affine):
    """inverse"""
    return [affine, warp]

def normalize(name='t1reg'):
    # too slow
    # normalize = Node(Registration(from_file="smri_ants_registration_settings.json"), name='ANTSRegistration')
    # refer to https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
    normalize = Node(Registration(), name=name)
    #normalize.inputs.interpolation = 'BSpline[<order=3>]'
    normalize.inputs.interpolation = 'Linear'
    normalize.inputs.use_histogram_matching = True ## within modality 1, across modality 0
    normalize.inputs.initial_moving_transform_com = 1 ## 1-match by center of mass [$t1brain,$template,1]
    normalize.inputs.dimension = 3
    #normalize.inputs.initial_moving_transform =
    normalize.inputs.winsorize_lower_quantile = 0.025
    normalize.inputs.winsorize_upper_quantile = 0.975
    normalize.inputs.collapse_output_transforms = True
    normalize.inputs.write_composite_transform = False # write one h5 file, hard to see
    normalize.inputs.float = False
    #normalize.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    #normalize.inputs.transform_parameters = [[0.1], [0.1], [0.1, 3, 0]]
    #normalize.inputs.metric = ['MI', 'MI', 'CC']
    #normalize.inputs.metric_weight = [1] * 3
    #normalize.inputs.sampling_strategy = ['Regular', 'Regular', None]
    #normalize.inputs.sampling_percentage = [0.25] * 3
    #normalize.inputs.radius_or_number_of_bins = [32, 32, 4]
    #normalize.inputs.convergence_window_size = [10] * 3
    #normalize.inputs.convergence_threshold = [1e-6] * 3
    #normalize.inputs.number_of_iterations = [[100, 70, 50, 30]] * 3
    #normalize.inputs.shrink_factors = [[8, 4, 2, 1]] * 3
    #normalize.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 3
    #normalize.inputs.sigma_units = ['vox'] * 3
    normalize.inputs.transforms = ['SyN']
    normalize.inputs.transform_parameters = [[0.1, 3, 0]]
    normalize.inputs.metric = ['CC']
    normalize.inputs.metric_weight = [1]
    normalize.inputs.sampling_strategy = [None]
    normalize.inputs.sampling_percentage = [None]
    normalize.inputs.radius_or_number_of_bins = [4]
    normalize.inputs.convergence_window_size = [5]
    normalize.inputs.convergence_threshold = [1e-6]
    normalize.inputs.number_of_iterations = [[100, 60, 50, 30]]
    normalize.inputs.shrink_factors = [[8, 4, 2, 1]]
    normalize.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    normalize.inputs.sigma_units = ['vox']
    normalize.inputs.num_threads = 8
    normalize.inputs.use_estimate_learning_rate_once = [True]
    normalize.config = {'execution':
                             {'remove_unnuecessary_outputs' : False}
                       }
    return normalize

def main():
    from bids.grabbids import BIDSLayout # not bids validated
    layout = BIDSLayout(input_dir)
    print(layout.get_subjects())
    print(layout.get_modalities())
    df = pd.read_table('info_T1_DTI.csv')
    df = df.loc[df.DTI_quality == 1, :]
    print(df.head())

    ## test 332 for hmc N4BiasFieldCorrection
    ## info.iterables = [('subject_id', ['332', '322'])]
    templates = {
                 'dti': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dti.nii.gz',
                 'bvals': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dti.bval',
                 'bvecs': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dti.bvec',
                 #'anat': 'sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz',
                }

    templates2 = {
                 'dki': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dki.nii.gz',
                 'kbvals': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dki.bval',
                 'kbvecs': 'sub-{subject_id}/dwi/sub-{subject_id}_run-1_dki.bvec',
    }

    #t1 = df.iloc[:, 1].map(lambda x: os.path.exists(os.path.join(input_dir, templates['anat'].format(subject_id=x))))
    dti = df.iloc[:, 1].map(lambda x: os.path.exists(os.path.join(input_dir, templates['dti'].format(subject_id=x))))
    bvals = df.iloc[:, 1].map(lambda x: os.path.exists(os.path.join(input_dir, templates['bvals'].format(subject_id=x))))
    bvecs = df.iloc[:, 1].map(lambda x: os.path.exists(os.path.join(input_dir, templates['bvecs'].format(subject_id=x))))

    #dki = df.iloc[:, 1].map(lambda x: os.path.exists(os.path.join(input_dir, templates2['dti'].format(subject_id=x))))
    #kbvals = df.iloc[:, 1].map(lambda x: os.path.exists(os.path.join(input_dir, templates2['kbvals'].format(subject_id=x))))
    #kbvecs = df.iloc[:, 1].map(lambda x: os.path.exists(os.path.join(input_dir, templates2['kbvecs'].format(subject_id=x))))
    
    #dfk = df.loc[dki & kbvals & kbvecs, :]
    df = df.loc[dti & bvals & bvecs, :]
    print(df.shape)

    #subjects = df.iloc[:2, 0]
    subjects = df.iloc[:25, 1]
    info = Node(IdentityInterface(fields=['subject_id']), name='subject_id')
    info.iterables = [('subject_id', subjects[:].unique())]

    sf = Node(SelectFiles(templates), name='selectfiles')
    sf.inputs.base_directory = input_dir
    sinker = Node(DataSink(), name='datasink')
    sinker.inputs.base_directory = output_dir
    sinker.inputs.substitutions = [('_subject_id_', 'subject_id')]   # do not need

    denoise_anat = Node(DenoiseImage(), name='denoise_anat', mem_gb=2)
    denoise_anat.inputs.dimension = 3

    ## anat specific
    reorient = Node(Reorient2Std(), name="reorient")
    enhan = Node(Function(function=enhance,
                          input_names=['in_file'],
                          output_names=['out_file']), name="enhance")
    enhan.inputs.clip_limit = 0.015

    #preprocess dwis
    dwi_workflow = ecc_hmc_pipeline(name='dwi_workflow')
    #dwi_workflow.base_dir = output_dir

    norm_T1 = normalize('t1reg')
    norm_T1.inputs.fixed_image = conf['jhu_t1']

    norm_b0 = normalize('b0reg')
    norm_b0.inputs.fixed_image = conf['jhu_b0']

    atfwd = MapNode(ApplyTransforms(), name='tranform', iterfield=['input_image'])
    atfwd.inputs.reference_image = conf['jhu_b0']
    atfwd.inputs.interpolation = 'Linear'
    atfwd.inputs.invert_transform_flags = [False, False]

    atinv = Node(ApplyTransforms(), name='invtranform')
    atinv.inputs.input_image = conf['jhu_atlas']
    atinv.inputs.interpolation = 'NearestNeighbor' # GenericLabel
    atinv.inputs.invert_transform_flags = [True, False]

    def reverse_transform(fwd):
        return fwd[::-1]

    #test denoise
    denoised = True

    preprocess_T1 = Workflow(name='preprocessT1')
    #Normalization to neonate template
    preprocess_T1.base_dir = output_dir

    jac = Node(CreateJacobianDeterminantImage(), name='jac')
    jac.inputs.imageDimension = 3
    jac.inputs.doLogJacobian = True
    jac.inputs.useGeometric = True

    def get_jac_output(i):
        return i.replace('.nii.gz', '_logj.nii.gz')
    def get_warped_output(oid):
        return "%s_warped.nii.gz" % oid

    select_warp = Node(Select(index=[1]), name='Reference')
    #preprocess_T1.connect([
    #    (info, sf, [('subject_id', 'subject_id')]),
    #    (sf, denoise_anat,       [('anat', 'input_image')]),
    #    (denoise_anat, reorient, [('output_image', 'in_file')]),
    #    (reorient, enhan,        [('out_file', 'in_file')]),
    #    (enhan,    sinker,       [('out_file', 'T1')]),
    #    (enhan,    norm_T1,      [('out_file', 'moving_image')]),
    #    (info,     norm_T1,      [(('subject_id', get_container), 'output_transform_prefix')]),
    #    (enhan,    atinv,        [('out_file', 'reference_image')]),
    #    (info,     norm_T1,      [(('subject_id', get_warped_output), 'output_warped_image')]),

    #    (norm_T1,  atinv,        [('reverse_transforms', 'transforms')]),
    #    (norm_T1,  select_warp,  [('reverse_transforms', 'inlist')]),

    #    (select_warp, jac,       [('out', 'deformationField'), (('out', get_jac_output), 'outputImage')]),
    #    (atinv,      sinker,     [('output_image', 't1_atlas')]),

    #    (norm_T1,  sinker,       [('warped_image', 'warped_t1'), ('forward_transforms', 'fwd_t1'), ('reverse_transforms', 'inv_t1')]),
    #    (jac,      sinker,       [('jacobian_image',  't1_bravo_jac')]),
    #])

    def gather_func(in_file1, in_file2):
        return [in_file1, in_file2]

    gather_dti = Node(Function(function=gather_func,
                     input_names=['in_file1', 'in_file2'],
                     output_names=['out_file']), name="gather_dti")

    preprocess_DTI = Workflow(name='preprocessDTI')
    #Normalization to neonate template
    preprocess_DTI.base_dir = output_dir
    preprocess_DTI.connect([
        (info, sf,               [('subject_id', 'subject_id')]),
        (sf, dwi_workflow,       [('bvecs',         'inputnode.in_bvec'),
                                  ('bvals',         'inputnode.in_bval'),
                                  ('dti',           'inputnode.in_file')]),

        (dwi_workflow, norm_b0,  [('outputnode.out_b0', 'moving_image')]),
        (info, norm_b0, [(('subject_id', get_container), 'output_transform_prefix'), 
                         (('subject_id', get_warped_output), 'output_warped_image')]),

        (dwi_workflow, sinker,   [('outputnode.out_b0', 'out_b0')]),

        (info, dwi_workflow,     [('subject_id',    'inputnode.subject')]),

        (dwi_workflow, sinker,   [('outputnode.out_denoised', 'out_denoised')]),
        (dwi_workflow, sinker,   [('outputnode.out_file', 'out_file')]),
        (dwi_workflow, sinker,   [('outputnode.out_mask', 'out_mask')]),

        (dwi_workflow, sinker,   [('outputnode.out_bvec', 'out_bvec')]),
        (dwi_workflow, sinker,   [('outputnode.out_jacobian', 'out_jacobian')]),
        (dwi_workflow, sinker,   [('outputnode.out_coeff', 'out_coeff')]),
        (dwi_workflow, sinker,   [('outputnode.out_warp', 'out_warp')]),

        (dwi_workflow, sinker,   [('outputnode.out_FA', 'out_FA')]),
        (dwi_workflow, sinker,   [('outputnode.out_MD', 'out_MD')]),

        ## Transform FA, MD etc.
        (dwi_workflow, gather_dti, [('outputnode.out_FA', 'in_file1'), ('outputnode.out_MD', 'in_file2')]),
        (gather_dti,   atfwd,      [('out_file', 'input_image')]),
        (norm_b0,      atfwd,      [(('forward_transforms', reverse_transform), 'transforms')]),
        (norm_b0,      sinker,     [('warped_image', 'warped_b0'), ('forward_transforms', 'fwd_b0'), ('reverse_transforms', 'inv_b0')]),
        (atfwd,        sinker,     [('output_image', 'dti_results')]),

        (info,         sinker,     [(('subject_id', get_container), 'container')]) # create sub-folder named as _subject_id_xxx, not necessary
    ])

    #use Nipype internal monitor of the submitted jobs
    #MultiProc always pending too long at the last task
    #because of multiprocessing module issues, expected fix at nipype-2.0
    #reference: https://github.com/nipy/nipype/issues/2548
    #try few n_procs, and mem_gb in Node()
    #preprocess_T1.write_graph(graph2use='colored', dotfilename='./t1.dot')
    #preprocess_T1.run('MultiProc', plugin_args={'n_procs': 16, 'raise_insufficient':True, 'memory_gb': 48})
    #preprocess_T1.run('SGEGraph', plugin_args={'dont_resubmit_completed_jobs': True,
    #                                            'qsub_args': ' -q normal.q ',
    #                                            'template':  'template.sh'})

    #preprocess_DTI.write_graph(graph2use='colored', dotfilename='./dti.dot')
    ##preprocess_DTI.run('MultiProc', plugin_args={'n_procs': 16, 'raise_insufficient':True, 'memory_gb': 48})

    #preprocess_DTI.run()
    preprocess_DTI.run('MultiProc', plugin_args={'n_procs': 36, 'raise_insufficient':True, 'memory_gb': 62})
    #preprocess_DTI.run('SGEGraph', plugin_args={'dont_resubmit_completed_jobs': True,
    #                                            'qsub_args': ' -q normal.q ',
    #                                            'template':  'template.sh'})

    # plugin IPython error, ipyparallel.error.RemoteError: IndexError(msg pending: 'f3fc415e-9f826b74f1881a4c278e971b')
    # cannot fix yet
    # preprocess.run('IPython', plugin_args={'n_procs': 16})

    #SGE sometimes crash for waiting too many queues, don't know why yet
    #error /data1/qinqian/MRI/hyperbilirubinemia/output/preprocess/dwi_workflow/_subject_id_322/denoise missing directory fix with template.sh
    #without customize denoise step, this can succeed...
    #succeed with two jobs tested, 76 jobs failed at DWIequalize problem, use SGEGraph instead
    #preprocess.run('SGE', plugin_args={'qsub_args': ' -cwd -q normal.q ',
    #                                   'template':  'template.sh',
    #                                   'overwrite': True})

    #use SGEGraph to monitor the submitted jobs

    #graph
    #preprocess.write_graph('colored')
    #pre.write_graph(graph2use='flat')

    #dtk.DTITracker
    #MultiProcPlugin

if __name__ == '__main__':
    main()

