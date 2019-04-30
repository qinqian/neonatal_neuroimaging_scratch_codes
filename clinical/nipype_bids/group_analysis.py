import glob
import nilearn
import os
from nipype.interfaces.ants import CreateJacobianDeterminantImage, ApplyTransforms
from nipype.interfaces.spm import MultipleRegressionDesign
from nipype.pipeline.engine import Node, MapNode, Workflow
from nipype import DataSink

def create_jac(input_image, fwd_warp, affine, output_dir):
    """ 
    Build template: https://github.com/CobraLab/documentation/wiki/ANTs-Multivariate-Template-Construction
    https://github.com/CobraLab/documentation/wiki/Deformation-Based-Morphometry-Analysis-using-ANTs-Multivariate-Template-Construction
    relative jacobian: only use inverse warp
    absolute jacobian: combine linear and nonlinear warp

    ANTS VBM: https://rpubs.com/antsr/phantomMorphometryStudy
       createjacobian geometry option, https://sourceforge.net/p/advants/discussion/840260/thread/84d24a38/

    FSL VBM:  struc_seg.pdf 38-51
    0-1: compression
    >1 : inflation
    """
    j = Workflow(name='jac')
    trans = MapNode(ApplyTransforms(), iterfield=['input_image', 'transforms', 'output_image'], name='absolute_deformable')

    trans.inputs.dimension = 3
    trans.inputs.input_image = input_image
    trans.inputs.transforms = list(zip(fwd_warp, affine))
    trans.inputs.output_image = list(map(lambda x: x.replace('Warp.nii.gz', 'compose.nii.gz'), fwd_warp))
    trans.synchronize = True

    trans.inputs.reference_image = '/data1/qinqian/MRI/Genetics/phaseB_get_yanglin_positive_ones/yanglin_genetics/fudan_neonate_cpvl/extract_template/template_T2_roi.nii.gz'
    trans.inputs.print_out_composite_warp_file = True
    trans.inputs.invert_transform_flags = [False, False]

    ajac = MapNode(CreateJacobianDeterminantImage(), iterfield=['deformationField', 'outputImage'], name='absolute_jac')
    ajac.inputs.imageDimension = 3
    ajac.inputs.doLogJacobian = False
    ajac.inputs.useGeometric = True
    ajac.synchronize = True
    ajac.inputs.outputImage = list(map(lambda x: x.replace('Warp.nii.gz', 'absj.nii.gz'), fwd_warp))
    # print('aaa',ajac.outputs)
    # print('bbb',trans.outputs)
  
    sinker = Node(DataSink(), name='datasink')
    sinker.inputs.base_directory = os.path.abspath(output_dir)

    j.connect([
        (trans, ajac, [('output_image', 'deformationField')]),
        (ajac,  sinker, [('jacobian_image', 'abs_jac')])
    ])

    jac = MapNode(CreateJacobianDeterminantImage(), iterfield=['deformationField', 'outputImage'], name='rel_jac')
    jac.inputs.imageDimension = 3
    jac.inputs.deformationField = fwd_warp
    jac.inputs.doLogJacobian = False
    jac.inputs.useGeometric = True
    jac.inputs.outputImage = list(map(lambda x: x.replace('.nii.gz', '_jac.nii.gz'), fwd_warp))
    jac.synchronize = True

    jac2 = MapNode(CreateJacobianDeterminantImage(), iterfield=['deformationField', 'outputImage'], name='log_jac')
    jac2.inputs.imageDimension = 3
    jac2.inputs.deformationField = fwd_warp
    jac2.inputs.doLogJacobian = True
    jac2.inputs.useGeometric = True
    jac2.inputs.outputImage = list(map(lambda x: x.replace('.nii.gz', '_logj.nii.gz'), fwd_warp))
    jac2.synchronize = True

    j.add_nodes([jac, jac2])

    j.run('MultiProc', plugin_args={'n_procs': 28})

def ants_group_analysis():
    """
    bigLMStats, p.adjust
    sccan
    rftResults (from SPM random field)
    """

def ants_roi_based_group_analysis():
    """
    """
    return

def nilearn_group_analysis():
    """ nilearn permute ols
    """



def afni():
    """@NoisySkullStrip"""
    return

def afni_group_analysis(groupA, groupB, clust, covariates):
    """ faster 
    3dttest++: must use bash instead of /bin/sh -> dash
         3dttest++ -setA ZZ_1.nii -setB ZZ_0.nii -prefix ZZtest.nii -no1sam -Clustsim 

    3dMEMA
    """
    return

def spm_group_analysis():
    """https://github.com/nipy/nipype/blob/master/examples/fmri_spm.py
    """
    model = MultipleRegressionDesign()

def fsl_group_analysis():
    """ randomise
    """
    return

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--list0', nargs='+')
    p.add_argument('--list', nargs='+')
    p.add_argument('--list2', nargs='+')
    p.add_argument('--output', type=str)
    args = p.parse_args()
    create_jac(args.list0, args.list, args.list2, args.output)

main()
