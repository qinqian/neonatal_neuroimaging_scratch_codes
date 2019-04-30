import numpy as np
from nipype.interfaces.ants import Registration, ApplyTransforms, N4BiasFieldCorrection
from nipype.interfaces.fsl import FLIRT, FAST, ExtractROI, BET
from nipype import Function
from nipype.interfaces.spm import DARTELNorm2MNI
from nipype.pipeline.engine import Node, MapNode, Workflow
from nipype import config, IdentityInterface, SelectFiles, DataSink
import os

#from bids.grabbids import BIDSLayout

template_directory = '/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalAtlas2'
t2_brain_template = '%s/template_T2.nii.gz' % template_directory
config.enable_debug_mode()

def get_template():
    templates = { 
                 't1':     '{group}/source/sub-{subject_id}/anat/sub-{subject_id}_run-01_T1w.nii.gz',
                 'flair':  '{group}/source/sub-{subject_id}/anat/sub-{subject_id}_run-01_T2w.nii.gz',
                 't2':     '{group}/source/sub-{subject_id}/anat/sub-{subject_id}_run-02_T2w.nii.gz',
                 'dwi':   '{group}/source/sub-{subject_id}/dwi/sub-{subject_id}_run-01_dwi.nii.gz',
                 'bvals': '{group}/source/sub-{subject_id}/dwi/sub-{subject_id}_run-01_dwi.bval',
                 #'bvecs': '{group}/source/sub-{subject_id}/dwi/sub-{subject_id}_run-01_dwi.bvec',
                 'adc':    '{group}/source/sub-{subject_id}/dwi/sub-{subject_id}_run-02_dwi.nii.gz'
                }
    return templates

def pipeline(bids_dir, subject_id, groups, output_id, output_dir):
    #segn4 = Node(FAST(), name='segment') # use Atropos later
    n4 = Node(N4BiasFieldCorrection(), name='N4')
    n4.inputs.dimension = 3
    n4.inputs.convergence_threshold = 1e-6
    n4.inputs.shrink_factor = 3
    n4.inputs.n_iterations = [50,50,30,20]
    #n4.inputs.n4.inputs.bspline_fitting_distance = 300

    multin4 = MapNode(N4BiasFieldCorrection(), name='multiN4', iterfield='input_image')
    multin4.inputs.dimension = 3
    multin4.inputs.convergence_threshold = 1e-6
    multin4.inputs.shrink_factor = 3
    multin4.inputs.n_iterations = [50,50,30,20]

    bet = Node(BET(), name='skullstrip')
    bet.inputs.frac = 0.2
    bet.inputs.mask = True

    # too slow 
    # normalize = Node(Registration(from_file="smri_ants_registration_settings.json"), name='ANTSRegistration')
    # refer to https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
    normalize = Node(Registration(), name='ANTSRegistration')
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
    normalize.inputs.number_of_iterations = [[100, 70, 60, 50]]
    normalize.inputs.shrink_factors = [[8, 4, 2, 1]]
    normalize.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    normalize.inputs.sigma_units = ['vox']
    normalize.inputs.num_threads = 10
    normalize.inputs.use_estimate_learning_rate_once = [True]

    normalize.config = {'execution':
                             {'remove_unnuecessary_outputs' : False}
                       }

    extract_template = Node(ExtractROI(), name='extract_template')
    extract_template.inputs.t_min=16  ## n - 1, n = 17, use 44 weeks
    extract_template.inputs.t_size=1
    extract_template.inputs.in_file = t2_brain_template
    tissue_list = ['csf.nii.gz', 'dgm.nii.gz', 'wm.nii.gz', 'brainmask.nii.gz',
                   'brainstem.nii.gz', 'cerebellum.nii.gz',  'cortex.nii.gz']
    tissue_list = list(map(lambda x: os.path.join(template_directory, x), tissue_list))

    extract_tissue = MapNode(ExtractROI(), name='extract_tissue', iterfield=['in_file'])
    extract_tissue.inputs.t_min=16  ## n - 1, n = 17, use 44 weeks
    extract_tissue.inputs.t_size=1
    extract_tissue.inputs.in_file = tissue_list

    #bids = BIDSLayout(bids_dir)
    #all_subject_id = bids.get_subjects()
    align = MapNode(FLIRT(), name='intra_align', iterfield=['in_file'])
    align.inputs.dof = 6

    info = Node(IdentityInterface(fields=['subject_id', 'group', 'output_id']), name='group_subject_id')
    info.iterables = [('subject_id', subject_id[:9]), ('group', groups[:9]), ('output_id', output_id[:9])]
    #info.iterables = [('subject_id', subject_id), ('group', groups), ('output_id', output_id)]
    info.synchronize = True # zip the subject id and group id

    sf = Node(SelectFiles(get_template()), name='selectfiles')
    sf.inputs.base_directory = bids_dir

    def get_index(bval):
        import numpy as np
        return np.where(np.loadtxt(bval) != 0)[0][0]

    def rename_uniform_id(t1, t2, adc, flair, dwi, sid, oid):
        import shutil
        shutil.copy(t1, t1.replace('sub-%s_run' % sid, 'sub-%s_run' % oid))
        shutil.copy(t2, t2.replace('sub-%s_run' % sid, 'sub-%s_run' % oid))
        shutil.copy(adc, adc.replace('sub-%s_run' % sid, 'sub-%s_run' % oid))
        shutil.copy(flair, flair.replace('sub-%s_run' % sid, 'sub-%s_run' % oid))
        shutil.copy(dwi, dwi.replace('sub-%s_run' % sid, 'sub-%s_run' % oid))
        return [t1.replace('sub-%s_run' % sid, 'sub-%s_run' % oid),  t2.replace('sub-%s_run' % sid, 'sub-%s_run' % oid),  adc.replace('sub-%s_run' % sid, 'sub-%s_run' % oid), dwi.replace('sub-%s_run' % sid, 'sub-%s_run' % oid)], flair.replace('sub-%s_run' % sid, 'sub-%s_run' % oid)

    ## change file name to uniform ids
    rename = Node(Function(input_names=['t1', 't2', 'adc', 'flair', 'dwi', 'sid', 'oid'],
                           #output_names=['t1_renamed', 't2_renamed', 'adc_renamed', 'flair_renamed'],
                           output_names=['agg', 'flair_renamed'],
                           function=rename_uniform_id), name='rename_data')

    sinker = Node(DataSink(), name='datasink')
    sinker.inputs.base_directory = os.path.abspath(output_dir)
    sinker.inputs.substitutions = [('_subject_id_', '_'), ('_group_', ''), ('_output_id_', '_')]   # do not need

    dwi_index = Node(Function(input_names=['bval'], output_names=['index'], function=get_index), name="dwi_index")
    extract_dwi = Node(ExtractROI(), name='extract_dwi')
    extract_dwi.inputs.t_size = 1
   
    def get_dwi(agg):
        return agg[-1]

    def add_dwi(agg, dwi):
        agg[-1] = dwi
        return agg

    def get_t2(modalities):
        return modalities[1]

    def get_warped_output(oid):
        return "%s_warped.nii.gz" % oid

    def get_needed_transform(agg, flair):
        agg[1] = flair
        return agg

    needed_transform = Node(Function(input_names=['agg', 'flair'], output_names=['agg_out'], function=get_needed_transform), name='get_needed_transform')

    add_modality = Node(Function(input_names=['agg', 'dwi'], output_names=['agg_out'], function=add_dwi), name='add_modality')
    atfwd = MapNode(ApplyTransforms(), name='tranform', iterfield=['input_image'])
    atfwd.inputs.interpolation = 'Linear'
    atfwd.inputs.invert_transform_flags = [False, False]

    atinv = MapNode(ApplyTransforms(), name='invtranform', iterfield=['input_image'])
    atinv.inputs.interpolation = 'Linear'
    atinv.inputs.invert_transform_flags = [True, False]

    def reverse_tranform(fwd):
        return fwd[::-1]

    w = Workflow(name='fudan_neonate_cpvl')
    w.base_dir = output_dir
    w.connect([

      (info, sf,        [('subject_id', 'subject_id'), ('group', 'group')]),
      (info, rename,    [('subject_id', 'sid'), ('output_id', 'oid')]),
      (sf,   rename,    [('t1', 't1'), ('flair', 'flair'), ('t2', 't2'), ('dwi', 'dwi'), ('adc', 'adc')]),
      (sf,   dwi_index, [('bvals', 'bval')]),
      (dwi_index, extract_dwi, [('index', 't_min')]),
      (rename, extract_dwi, [(('agg', get_dwi), 'in_file')]),

      (extract_dwi, add_modality, [('roi_file', 'dwi')]),
      (rename, add_modality,  [('agg', 'agg')]),

      (add_modality, multin4, [('agg_out', 'input_image')]),
 
      (rename,  n4,       [('flair_renamed', 'input_image')]),
      (n4,      align,    [('output_image', 'reference')]),
      (multin4, align,    [('output_image', 'in_file')]),
      (n4,      sinker,   [('output_image', 'flair')]),
      (align,   sinker,   [('out_file',     'align')]),

      (n4,      needed_transform,  [('output_image', 'flair')]),
      (align,   needed_transform,  [('out_file', 'agg')]),

      (align,   bet,      [(('out_file', get_t2), 'in_file')]),
      (bet,     sinker,   [('mask_file','mask'), ('out_file', 't2_strip')]),

      (extract_template, normalize, [('roi_file', 'fixed_image')]),
      (bet,  normalize,   [('out_file', 'moving_image')]),
      (info, normalize,   [('output_id', 'output_transform_prefix')]),
      (info, normalize,   [(('output_id', get_warped_output), 'output_warped_image')]),

      (needed_transform, atfwd,     [('agg_out', 'input_image')]),
      (extract_template, atfwd,         [('roi_file', 'reference_image')]),
      (normalize,        atfwd,         [(('forward_transforms', reverse_tranform), 'transforms')]),

      (bet,              atinv,         [('out_file', 'reference_image')]),
      (extract_tissue,   atinv,         [('roi_file', 'input_image')]),
      (normalize,        atinv,         [('reverse_transforms', 'transforms')]),

      (normalize, sinker, [('warped_image', 'neonate_template'),
                           ('forward_transforms', 'transform'),
                           ('reverse_transforms', 'invtransform')
                           #('composite_transform', 'transform'),
                           #('inverse_composite_transform', 'invtransform')
                          ]),
      (atfwd,     sinker, [('output_image', 'applytransformimage')]),
      (atinv,     sinker, [('output_image', 'applytranformtissue')])
    ])
    print(bids_dir)
    return w

