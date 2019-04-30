import os

from nipype import IdentityInterface, Function, Select
from nipype.interfaces.ants import DenoiseImage
from nipype.pipeline.engine import Node, MapNode, Workflow

from nipype.workflows.dmri.fsl.dti import create_eddy_correct_pipeline
from nipype.workflows.dmri.fsl.artifacts import ecc_pipeline, hmc_pipeline, remove_bias, _xfm_jacobian
import nipype.interfaces.diffusion_toolkit as dtk

from nipype.interfaces.fsl import Reorient2Std, ExtractROI, DTIFit, ImageMaths, Threshold, BinaryMaths, \
    Split, BET, ApplyWarp, ConvertWarp, WarpUtils, MultiImageMaths, Merge, ApplyXFM, FLIRT
from nipype.workflows.dmri.fsl.utils import b0_average, extract_bval, dwi_flirt, recompose_xfm, recompose_dwi, \
    time_avg, b0_indices

from mrtrix3 import dwidenoise


def get_mask(in_file):
    import os
    os.system('fLABEL {} -p 2'.format(in_file))
    return in_file.replace('.nii', '-strip.nii')


def get_container(subject_id):
    return str(subject_id)

def apply_parts_corrections(name='UnwarpArtifacts'):
    """
    from apply_all_corrections
    Combines two lists of linear transforms
    """
    inputnode = Node(
        IdentityInterface(fields=['in_hmc', 'in_ecc', 'in_dwi']),
        name='inputnode')
    outputnode = Node(
        IdentityInterface(
            fields=['out_file', 'out_warp', 'out_coeff', 'out_jacobian', 'mask_file', 'b0_reference']),
        name='outputnode')

    bet_dwi1 = Node(BET(frac=0.2, mask=True, robust=True), name='bet_dwi_post')

    warps = MapNode(
        ConvertWarp(relwarp=True),
        iterfield=['premat', 'postmat'],
        name='ConvertWarp')

    selref = Node(Select(index=[0]), name='Reference')
    selref2 = Node(Select(index=[0]), name='ReferenceIso')
    split = Node(Split(dimension='t'), name='SplitDWIs')

    unwarp = MapNode(
        ApplyWarp(),
        iterfield=['in_file', 'field_file'],
        name='UnwarpDWIs')

    coeffs = MapNode(
        WarpUtils(out_format='spline'),
        iterfield=['in_file'],
        name='CoeffComp')

    jacobian = MapNode(
        WarpUtils(write_jacobian=True),
        iterfield=['in_file'],
        name='JacobianComp')

    jacmult = MapNode(
        MultiImageMaths(op_string='-mul %s'),
        iterfield=['in_file', 'operand_files'],
        name='ModulateDWIs')

    thres = MapNode(
        Threshold(thresh=0.0),
        iterfield=['in_file'],
        name='RemoveNegative')

    iso = MapNode(FLIRT(), name='isotropic', iterfield=['in_file', 'reference'])
    iso.inputs.apply_isoxfm = True

    merge = Node(Merge(dimension='t'), name='MergeDWIs')

    wf = Workflow(name=name)
    wf.connect([
        (inputnode, warps, [
        ('in_hmc', 'premat'), ('in_ecc', 'postmat'), ('in_dwi', 'reference')
        ]),

        (inputnode, split, [('in_dwi', 'in_file')]),
        (split, selref, [('out_files', 'inlist')]),

        (warps, unwarp, [('out_file', 'field_file')]),
        (split, unwarp, [('out_files', 'in_file')]),
        (selref, unwarp, [('out', 'ref_file')]),

        (selref, coeffs, [('out', 'reference')]),
        (warps,  coeffs, [('out_file', 'in_file')]),

        (selref, jacobian, [('out', 'reference')]),
        (coeffs, jacobian, [('out_file', 'in_file')]),

        (unwarp, jacmult,   [('out_file', 'in_file')]),
        (jacobian, jacmult, [('out_jacobian', 'operand_files')]),

        (jacmult, thres, [('out_file', 'in_file')]),
        (thres,   iso,   [('out_file', 'in_file'), ('out_file', 'reference')]),
        (iso,     merge, [('out_file', 'in_files')]),

        ## second b0 mask and average b0 would cause N4 crash
        ## use b0 mask from first b-value = 0
        (iso,      selref2,     [('out_file', 'inlist')]),
        #(selref,   outputnode, [('out', 'b0_reference')]),

        (selref2,   bet_dwi1,   [('out', 'in_file')]),
        (bet_dwi1, outputnode,  [('mask_file', 'mask_file')]),
        (bet_dwi1, outputnode,  [('out_file',  'b0_reference')]), ## JHU b0 template is skull striped

        (warps, outputnode, [('out_file', 'out_warp')]),
        (coeffs, outputnode, [('out_file', 'out_coeff')]),
        (jacobian, outputnode, [('out_jacobian', 'out_jacobian')]),
        (merge, outputnode, [('merged_file', 'out_file')]),
    ])
    return wf

def simple_dwi_process(name='dwi_correction'):
    inputnode = Node(IdentityInterface(
        fields=['subject', 'in_file', 'in_bvec', 'in_bval']
    ), name='inputnode')

    outputnode = Node(IdentityInterface(
        fields=['out_file', 'out_mask', 'out_bvec', 'out_b0',
                'out_FA', 'out_MD']
    ), name='outputnode')

    eddy_correct = create_eddy_correct_pipeline(name='eddy_correct')

    wf = Workflow(name='dwi_proc')
    wf.connect([
        (inputnode, eddy_correct, [('subject', 'inputnode.subject')]), ##
    ])
    return wf


def ecc_hmc_pipeline(name='dwi_correction'):
    """
    merge hmc_pipeline, ecc_pipeline
    """
    inputnode = Node(IdentityInterface(
        fields=['subject', 'in_file', 'in_bvec', 'in_bval']
    ), name='inputnode')

    denoised = True
    if denoised:
        denoise = Node(dwidenoise(), name='denoise')
        #denoise = Node(DenoiseImage(), name='denoise', mem_gb=4) # too slow
        #denoise.inputs.dimension = 4
        outputnode = Node(IdentityInterface(
            fields=['out_file', 'out_mask', 'out_bvec', 'out_b0', 'out_warp', 'out_coeff', 'out_jacobian',
                    'out_FA', 'out_MD', 'out_denoised']
        ), name='outputnode')
    else:
        outputnode = Node(IdentityInterface(
            fields=['out_file', 'out_mask', 'out_bvec', 'out_b0', 'out_warp', 'out_coeff', 'out_jacobian',
                    'out_FA', 'out_MD']
        ), name='outputnode')

    #denoise = Node(Function(function=mrtrix3_denoise,
    #                        input_names=['in_file'],
    #                        output_names=['out_file']),
    #                        name='denoise')

    def b0_0(in_bval, max_b=10.0):
        """
        Extract the first indices with b-value = 0
        replace b0_indices,
        NOTE: taking the average b0 to generate mask image
              would lead to the crash of N4BiasFieldCorrection in hmc and ecc
              details: physical space difference between mask image and any dwi larger than 1e-6
        """
        import numpy as np
        bval = np.loadtxt(in_bval)
        return [np.argwhere(bval <= max_b).flatten().tolist()[0]]


    list_b0  = Node(Function(
        input_names=['in_bval'],
        output_names=['out_idx'],
        function=b0_indices), name = 'list_b0')

    avg_b0_0 = Node(Function(
        input_names=['in_file', 'index'],
        output_names=['out_file'],
        function=time_avg), name='average_b0_0')


    bet_dwi0 = Node(BET(frac=0.2, mask=True, robust=True), name='bet_dwi_pre')

    hmc = hmc_pipeline()
    ecc = ecc_pipeline()

#   apply_warp = apply_all_corrections()
    apply_warp = apply_parts_corrections()    #without phase encoding warps1

    #simple ways
    #eddy_correct = create_eddy_correct_pipeline(name='eddycorrect')
    #eddy_correct.inputs.input_node.ref_num = 0

    dtifit = Node(DTIFit(), name="dtifit")
    #print(dtifit.outputs)

    pipeline = Workflow(name=name)
    if denoised:
        pipeline.connect([
            # denoise
            (inputnode, denoise,         [('in_file',   'in_file')]),
            #(inputnode, denoise,         [('in_file',   'input_image')]), # ants
            #(inputnode, hmc,         [('in_file',   'inputnode.in_file')]),

            ## initial b0 mask with first b-value = 0
            (inputnode, list_b0,         [('in_bval',   'in_bval')]),
            (denoise,   avg_b0_0,        [('out_file',  'in_file')]),
            #(denoise,   avg_b0_0,        [('output_image',  'in_file')]),
            #(inputnode, avg_b0_0,        [('in_file',  'in_file')]),
            (list_b0,   avg_b0_0,        [('out_idx',   'index')]),
            (avg_b0_0,  bet_dwi0,        [('out_file',  'in_file')]),

            ## head motion correction
            (denoise,   hmc,             [('out_file',  'inputnode.in_file')]),
            #(denoise,   hmc,             [('output_image',  'inputnode.in_file')]),
            (bet_dwi0,  hmc,             [('mask_file', 'inputnode.in_mask')]),
            (inputnode, hmc,             [('in_bvec', 'inputnode.in_bvec'),
                                          ('in_bval', 'inputnode.in_bval')]),

            ## eddy correction hmc=>ecc
            (hmc, ecc,                   [('outputnode.out_xfms', 'inputnode.in_xfms')]),
            (inputnode, ecc,             [('in_bval', 'inputnode.in_bval')]),
            (denoise, ecc,               [('out_file', 'inputnode.in_file')]),
            #(denoise, ecc,               [('output_image', 'inputnode.in_file')]),
            #(inputnode, ecc,               [('in_file', 'inputnode.in_file')]),
            (bet_dwi0, ecc,              [('mask_file', 'inputnode.in_mask')]),


            ## apply both hmc and ecc transformation
            (denoise,   apply_warp,      [('out_file',            'inputnode.in_dwi')]),
            #(denoise,   apply_warp,      [('output_image',            'inputnode.in_dwi')]),
            #(inputnode,   apply_warp,      [('in_file',            'inputnode.in_dwi')]),
            (hmc,       apply_warp,      [('outputnode.out_xfms', 'inputnode.in_hmc')]),
            (ecc,       apply_warp,      [('outputnode.out_xfms', 'inputnode.in_ecc')]),

            ## dtifit to get FA and MD
            (apply_warp, dtifit,         [('outputnode.out_file',     'dwi')]),
            (hmc,        dtifit,         [('outputnode.out_bvec',     'bvecs')]),
            (inputnode,  dtifit,         [('in_bval',                 'bvals')]),
            (apply_warp, dtifit,         [('outputnode.mask_file',    'mask')]),
            (inputnode,  dtifit,         [(('subject', get_container),'base_name')]),

            (denoise,    outputnode,     [('out_file', 'out_denoised')]),
            #(denoise,    outputnode,     [('output_image', 'out_denoised')]),
            (dtifit,     outputnode,     [('FA', 'out_FA'),
                                          ('MD', 'out_MD')]),
            (hmc,       outputnode,      [('outputnode.out_bvec', 'out_bvec')]), # rotated b vecs
            (apply_warp, outputnode,     [('outputnode.out_file', 'out_file'),
                                          ('outputnode.out_warp', 'out_warp'),
                                          ('outputnode.out_coeff', 'out_coeff'),
                                          ('outputnode.mask_file', 'out_mask'),
                                          ('outputnode.b0_reference', 'out_b0'),
                                          ('outputnode.out_jacobian', 'out_jacobian')])
        ])
    else:
        pipeline.connect([
            (inputnode, hmc,         [('in_file',   'inputnode.in_file')]),

            ## initial b0 mask with first b-value = 0
            (inputnode, list_b0,         [('in_bval',   'in_bval')]),
            (inputnode, avg_b0_0,        [('in_file',  'in_file')]),
            (list_b0,   avg_b0_0,        [('out_idx',   'index')]),
            (avg_b0_0,  bet_dwi0,        [('out_file',  'in_file')]),

            ## head motion correction
            (bet_dwi0,  hmc,             [('mask_file', 'inputnode.in_mask')]),
            (inputnode, hmc,             [('in_bvec', 'inputnode.in_bvec'),
                                          ('in_bval', 'inputnode.in_bval')]),

            ## eddy correction hmc=>ecc
            (hmc, ecc,                   [('outputnode.out_xfms', 'inputnode.in_xfms')]),
            (inputnode, ecc,             [('in_bval', 'inputnode.in_bval')]),
            (inputnode, ecc,               [('in_file', 'inputnode.in_file')]),
            (bet_dwi0, ecc,              [('mask_file', 'inputnode.in_mask')]),

            ## apply both hmc and ecc transformation
            (inputnode,   apply_warp,      [('in_file',            'inputnode.in_dwi')]),
            (hmc,       apply_warp,      [('outputnode.out_xfms', 'inputnode.in_hmc')]),
            (ecc,       apply_warp,      [('outputnode.out_xfms', 'inputnode.in_ecc')]),

            ## dtifit to get FA and MD
            (apply_warp, dtifit,         [('outputnode.out_file',     'dwi')]),
            (hmc,        dtifit,         [('outputnode.out_bvec',     'bvecs')]),
            (inputnode,  dtifit,         [('in_bval',                 'bvals')]),
            (apply_warp,   dtifit,       [('outputnode.mask_file',               'mask')]),
            (inputnode,  dtifit,         [(('subject', get_container),'base_name')]),

            (dtifit,     outputnode,     [('FA', 'out_FA'),
                                          ('MD', 'out_MD')]),
            (hmc,       outputnode,      [('outputnode.out_bvec', 'out_bvec')]), # rotated b vecs
            (apply_warp, outputnode,     [('outputnode.out_file', 'out_file'),
                                          ('outputnode.out_warp', 'out_warp'),
                                          ('outputnode.out_coeff', 'out_coeff'),
                                          ('outputnode.out_jacobian', 'out_jacobian')])
        ])

    return pipeline
