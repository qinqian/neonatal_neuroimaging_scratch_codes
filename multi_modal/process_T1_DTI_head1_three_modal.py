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
from nipype.workflows.dmri.fsl.utils import b0_average, extract_bval, dwi_flirt, recompose_xfm, recompose_dwi, \
    time_avg, b0_indices
from nipype.interfaces.fsl import Reorient2Std, ExtractROI, DTIFit, ImageMaths, FLIRT, BET

from nipype.workflows.fmri.fsl import create_featreg_preproc
from nipype.interfaces.fsl import FAST # accept T1, T2, proton
from nipype.interfaces import fsl
from nipype.pipeline.plugins import MultiProcPlugin
from nipype.utils.filemanip import makedirs
from nipype.workflows.dmri.fsl.utils import enhance
from workflows import ecc_hmc_pipeline, get_container
import argparse

fsl.FSLCommand.set_default_output_type("NIFTI_GZ")

#from nipype.caching import Memory
#mem = Memory('.')
from nipype import config, logging


def agg_transforms(warp, affine):
    """forward"""
    return [warp, affine]

def inv_agg_transforms(warp, affine):
    """inverse"""
    return [affine, warp]

def normalize(name='t1reg', type='SyN'):
    # too slow
    # normalize = Node(Registration(from_file="smri_ants_registration_settings.json"), name='ANTSRegistration')
    # refer to https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
    normalize = Node(Registration(), name=name)
    #normalize.inputs.interpolation = 'BSpline[<order=3>]'
    normalize.inputs.interpolation = 'Linear'
    normalize.inputs.use_histogram_matching = True ## within modality 1, across modality 0
    normalize.inputs.initial_moving_transform_com = 1 ## 1-match by center of mass [$t1brain,$template,1]
    normalize.inputs.dimension = 3
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
    normalize.inputs.transforms = [type]
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
    normalize.inputs.num_threads = 1
    normalize.inputs.use_estimate_learning_rate_once = [True]
    normalize.config = {'execution':
                             {'remove_unnuecessary_outputs' : False}
                       }
    return normalize

def process_dti(conf, output_dir, info, sf, sinker):
    """ process DTI to FA and MD """
    #preprocess dwis
    dwi_workflow = ecc_hmc_pipeline(name='dwi_workflow', denoised=False)
    norm_b0 = normalize('b0reg')
    norm_b0.inputs.fixed_image = conf['jhu_b0']

    atfwd = MapNode(ApplyTransforms(), name='tranform', iterfield=['input_image'])
    atfwd.inputs.reference_image = conf['jhu_b0']
    atfwd.inputs.interpolation = 'Linear'
    atfwd.inputs.invert_transform_flags = [False, False]

    def reverse_transform(fwd):
        return fwd[::-1]

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
        #(dwi_workflow, norm_b0,  [('outputnode.out_b0', 'moving_image')]),
        #(info, norm_b0, [(('subject_id', get_container), 'output_transform_prefix'), 
        #                 (('subject_id', get_warped_output), 'output_warped_image')]),
        (dwi_workflow, sinker,   [('outputnode.out_b0', 'out_b0')]),
        (info, dwi_workflow,     [('subject_id',    'inputnode.subject')]),
        #(dwi_workflow, sinker,   [('outputnode.out_denoised', 'out_denoised')]),
        (dwi_workflow, sinker,   [('outputnode.out_file', 'out_file')]),
        (dwi_workflow, sinker,   [('outputnode.out_mask', 'out_mask')]),
        (dwi_workflow, sinker,   [('outputnode.out_bvec', 'out_bvec')]),
        (dwi_workflow, sinker,   [('outputnode.out_jacobian', 'out_jacobian')]),
        (dwi_workflow, sinker,   [('outputnode.out_coeff', 'out_coeff')]),
        (dwi_workflow, sinker,   [('outputnode.out_warp', 'out_warp')]),
        (dwi_workflow, sinker,   [('outputnode.out_FA', 'out_FA')]),
        (dwi_workflow, sinker,   [('outputnode.out_MD', 'out_MD')]),

        ## Transform FA, MD etc.
        #(dwi_workflow, gather_dti, [('outputnode.out_FA', 'in_file1'), ('outputnode.out_MD', 'in_file2')]),
        #(gather_dti,   atfwd,      [('out_file', 'input_image')]),
        #(norm_b0,      atfwd,      [(('forward_transforms', reverse_transform), 'transforms')]),
        #(norm_b0,      sinker,     [('warped_image', 'warped_b0'), ('forward_transforms', 'fwd_b0'), ('reverse_transforms', 'inv_b0')]),
        #(atfwd,        sinker,     [('output_image', 'dti_results')]),
        (info,         sinker,     [(('subject_id', get_container), 'container')]) # create sub-folder named as _subject_id_xxx, not necessary
    ])
    return preprocess_DTI

def get_jac_output(i):
    return i.replace('.nii.gz', '_logj.nii.gz')

def get_warped_output(oid):
    return "%s_warped.nii.gz" % oid

def process_t1(conf, output_dir, info, sf, sinker):
    preprocess_T1 = Workflow(name='preprocessT1')
    preprocess_T1.base_dir = output_dir
    denoise_anat = Node(DenoiseImage(), name='denoise_anat', mem_gb=2)
    denoise_anat.inputs.dimension = 3
    reorient = Node(Reorient2Std(), name="reorient")
    enhan = Node(Function(function=enhance,
                          input_names=['in_file'],
                          output_names=['out_file']), name="enhance")
    enhan.inputs.clip_limit = 0.015

    bet_t1 = Node(BET(frac=0.1, mask=True, robust=False, args=' -B '), name='bet_t1')
    #bet_b0 = Node(BET(frac=0.2, mask=True, robust=True), name='bet_b0')
    #norm_T1 = normalize('t1reg')
    #norm_T1.inputs.fixed_image = conf['jhu_t1']

    #iso = Node(FLIRT(), name='isotropic_sample')
    #iso.inputs.apply_isoxfm = True

    #align_b0 = Node(FLIRT(bins=640, cost_func='mutualinfo', args=' -coarsesearch 90 -finesearch 90 '), name='align_t1_to_b0')
    #align_b0.inputs.dof = 6
    #align_b0.inputs.searchr_x = [-30, 40]
    #align_b0.inputs.searchr_y = [-30, 40]
    #align_b0.inputs.searchr_z = [-30, 40]
    #align_b0 = normalize('align_b0', 'Rigid')

    jac = Node(CreateJacobianDeterminantImage(), name='jac')
    jac.inputs.imageDimension = 3
    jac.inputs.doLogJacobian = True
    jac.inputs.useGeometric = True

    atinv = Node(ApplyTransforms(), name='invtranform')
    atinv.inputs.input_image = conf['jhu_atlas']
    atinv.inputs.interpolation = 'NearestNeighbor' # GenericLabel
    atinv.inputs.invert_transform_flags = [True, False]

    list_b0  = Node(Function(
        input_names=['in_bval'],
        output_names=['out_idx'],
        function=b0_indices), name = 'list_b0')
    avg_b0_0 = Node(Function(
        input_names=['in_file', 'index'],
        output_names=['out_file'],
        function=time_avg), name='average_b0_0')

    select_warp = Node(Select(index=[1]), name='Reference')
    preprocess_T1.connect([
        (info, sf,               [('subject_id', 'subject_id')]),
        #(sf, denoise_anat,       [('anat', 'input_image')]),
        #(sf, list_b0,            [('bvals', 'in_bval')]),
        #(sf, avg_b0_0,           [('dti', 'in_file')]),
        #(list_b0, avg_b0_0,      [('out_idx', 'index')]),
        #(denoise_anat, reorient, [('output_image', 'in_file')]),
        #(reorient, enhan,        [('out_file', 'in_file')]),
        (sf,       bet_t1,        [('anat', 'in_file')]),
        #(enhan,    sinker,        [('out_file', 'T1')]),
        #(avg_b0_0, bet_b0,       [('out_file', 'in_file')]),
        #(bet_b0,   iso,          [('out_file', 'in_file'), ('out_file', 'reference')]),
        #(iso, align_b0,          [('out_file', 'reference')]),
        #(iso, align_b0,          [('out_file', 'fixed_image')]),
        #(enhan,    bet_t1,       [('out_file', 'in_file')]),
        #(bet_t1,   align_b0,     [('out_file', 'in_file')]),
        #(bet_t1,   align_b0,     [('out_file', 'moving_image')]),
        #(align_b0, sinker,       [('out_file', 'aligned')]),
        #(align_b0, sinker,       [('warped_image', 'aligned')]),
        (bet_t1, denoise_anat,    [('out_file', 'input_image')]),
        (denoise_anat, enhan,     [('output_image', 'in_file')]),
        (enhan, sinker,           [('out_file', 'bet_t1')]),
        (info, sinker,            [(('subject_id', get_container), 'container')])
        #(enhan,    norm_T1,      [('out_file', 'moving_image')]),
        #(info,     norm_T1,      [(('subject_id', get_container), 'output_transform_prefix')]),
        #(enhan,    atinv,        [('out_file', 'reference_image')]),
        #(info,     norm_T1,      [(('subject_id', get_warped_output), 'output_warped_image')]),
        #(norm_T1,  atinv,        [('reverse_transforms', 'transforms')]),
        #(norm_T1,  select_warp,  [('reverse_transforms', 'inlist')]),
        #(select_warp, jac,       [('out', 'deformationField'), (('out', get_jac_output), 'outputImage')]),
        #(atinv,      sinker,     [('output_image', 't1_atlas')]),
        #(norm_T1,  sinker,       [('warped_image', 'warped_t1'), ('forward_transforms', 'fwd_t1'), ('reverse_transforms', 'inv_t1')]),
        #(jac,      sinker,       [('jacobian_image',  't1_bravo_jac')]),
    ])
    return preprocess_T1


def get_input(meta):
    df = pd.read_table(meta)
    templates = {
                 'dti': '{subject_id}/dwi/{subject_id}_run-1_dti.nii.gz',
                 'bvals': '{subject_id}/dwi/{subject_id}_run-1_dti.bval',
                 'bvecs': '{subject_id}/dwi/{subject_id}_run-1_dti.bvec', 
                 'anat': '{subject_id}/anat/{subject_id}_T1w.nii.gz',
                 'bold': '{subject_id}/func/{subject_id}_run-1_bold.nii.gz'
                }
    #subjects = df.iloc[:, ].ID
    subjects = df.iloc[:1, ].ID
    subjects = subjects[:].unique()
    subjects = subjects[subjects!=159] # when add mask to N4bias program, it fails... bold is wrong...
    #subjects = subjects[subjects!=156]
    return templates, subjects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t1', action='store_true', default=False)
    parser.add_argument('-dti', action='store_true', default=False)
    parser.add_argument('-bold', default='', type=str)
    parser.add_argument('-meta', default='kai_2017_dti_t1_bold_to_be_processed.xls', type=str)
    parser.add_argument('-output', default='/cluster/home/qinqian/mri/hyperbilirubinemia/output', type=str)
    parser.add_argument('-input', default='/cluster/home/qinqian/mri/hyperbilirubinemia/', type=str)
    args = parser.parse_args()
   
    if args.bold != '':
        template_data = {'T1': '{subject_id}_T1w_noise_corrected_reoriented_enh.nii.gz',
                         'b0': '{subject_id}_b0_bet.nii.gz',
                         'bold': '{subject_id}_run-1_bold.nii.gz'}
        data_dir = pd.read_table(args.bold, delimiter=' ', header=None)
        info = Node(IdentityInterface(fields=['subject_id']), name='subject_id')
        subjects = data_dir.iloc[:, 0].unique()
        subjects = subjects[subjects!=159] # when add mask to N4bias program, it fails... bold is wrong...
        info.iterables = [('subject_id', subjects[80:])]
        sf = Node(SelectFiles(template_data), name='selectfiles')
        sf.inputs.base_directory = '/data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/'
        sinker = Node(DataSink(), name='datasink')
        sinker.inputs.base_directory = '/data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/'

        preprocess_bold = Workflow(name='process_bold')
        preprocess_bold.base_dir = '/data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/nipype'
        extract = Node(ExtractROI(t_min=5, t_size=-1, output_type='NIFTI_GZ'), name="extract")
        preproc = create_featreg_preproc(highpass=False)
        preproc.inputs.inputspec.fwhm = 4.5
        #preproc.base_dir = 'results/nipype'
        preprocess_bold.connect(info, 'subject_id', sf, 'subject_id')
        preprocess_bold.connect(sf, 'bold', extract, 'in_file')
        preprocess_bold.connect(extract, 'roi_file', sinker, 'remove_5vol')
        preprocess_bold.connect(extract, 'roi_file', preproc, 'inputspec.func')
        preprocess_bold.connect(preproc, 'outputspec.smoothed_files', sinker, 'smooth_cleaned_bold')
        preprocess_bold.connect(preproc, 'outputspec.mean', sinker, 'cleaned_bold/average_bold')
        preprocess_bold.connect(preproc, 'outputspec.motion_plots', sinker, 'cleaned_bold/plot_motion')
        preprocess_bold.connect(preproc, 'outputspec.motion_parameters', sinker, 'cleaned_bold/motion_par')
        preprocess_bold.connect(preproc, 'outputspec.mask', sinker, 'cleaned_bold/bold_mask')
        preprocess_bold.write_graph(graph2use='colored')
        preprocess_bold.run('MultiProc', plugin_args={'n_procs': 16, 'raise_insufficient':True, 'memory_gb': 48})
        #t1_workflow.run('PBSGraph', plugin_args={'dont_resubmit_completed_jobs': True,
        #                                       'qsub_args': ' -q q2 ',
        #                                       'template':  'template.head1'})
        return

    input_dir = args.input
    output_dir = args.output
    config.enable_debug_mode()
    config.set('execution', 'remove_unnecessary_outputs', 'true')
    config.set('execution', 'keep_unnecessary_outputs', 'false')
    config.set('execution', 'job_finished_timeout', '20')
    config.set('execution', 'crashdump_dir', os.path.join(output_dir, 'crash'))
    config.set('execution', 'stop_on_first_crash', 'true')
    logging.update_logging(config)
    with open('template.json') as j:
        conf = json.loads(j.read())
    templates, subjects = get_input(args.meta)
    info = Node(IdentityInterface(fields=['subject_id']), name='subject_id')
    info.iterables = [('subject_id', subjects)]
    sf = Node(SelectFiles(templates), name='selectfiles')
    sf.inputs.base_directory = input_dir
    sinker = Node(DataSink(), name='datasink')
    sinker.inputs.base_directory = output_dir
    sinker.inputs.substitutions = [('_subject_id_', 'subject_id')]   # do not need

    if args.t1:
        t1_workflow = process_t1(conf, output_dir, info, sf, sinker)
        #use Nipype internal monitor of the submitted jobs
        #MultiProc always pending too long at the last task
        #because of multiprocessing module issues, expected fix at nipype-2.0
        #reference: https://github.com/nipy/nipype/issues/2548
        #try few n_procs, and mem_gb in Node()
        #t1_workflow.write_graph(graph2use='colored', dotfilename='./t1.dot')
        t1_workflow.run('PBSGraph', plugin_args={'dont_resubmit_completed_jobs': True,
                                               'qsub_args': ' -q q2 ',
                                               'template':  'template.head1'})
        #t1_workflow.run('MultiProc', plugin_args={'n_procs': 16, 'raise_insufficient':True, 'memory_gb': 48})
        #t1_workflow.run('SGEGraph', plugin_args={'dont_resubmit_completed_jobs': True,
        #                                            'qsub_args': ' -q normal.q ',
        #                                            'template':  'template.sh'})
        return

    if args.dti:
        dti_workflow = process_dti(conf, output_dir, info, sf, sinker)
        dti_workflow.write_graph(graph2use='colored', dotfilename='./dti.dot')
        #dti_workflow.run('MultiProc', plugin_args={'n_procs': 16, 'raise_insufficient':True, 'memory_gb': 48})
        #dti_workflow.run('PBSGraph', plugin_args={'dont_resubmit_completed_jobs': True,
        #                                            'qsub_args': ' -q q2 ',
        #                                            'template':  'template.head1'})
        #dti_workflow.run('SGEGraph', plugin_args={'dont_resubmit_completed_jobs': True,
        #                                            #'qsub_args': ' -q normal.q ',
        #                                            'qsub_args': ' -q q1 ',
        #                                            'template':  'template.head1'})

if __name__ == '__main__':
    main()
