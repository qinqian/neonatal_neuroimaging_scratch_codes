import os
from nipype.interfaces.ants.base import ANTSCommandInputSpec, ANTSCommand
from niftynet.io.misc_io import split_filename
from nipype.interfaces.freesurfer import FixTopology, Sphere
from nipype.interfaces.fsl import FSLCommand
from nipype.interfaces.fsl.base import FSLCommandInputSpec

from nipype.interfaces.base import (
    TraitedSpec,
    CommandLineInputSpec,
    CommandLine,
    File,
    isdefined, DynamicTraitedSpec)
from nipype.utils.filemanip import makedirs


def mrtrix3_denoise(in_file):
    import os
    out_file = in_file.replace('.nii.gz', '_denoise.nii.gz')
    os.system('dwidenoise {0} {1}'.format(in_file, out_file))
    return out_file


#class MRtrix3DenoiseInputSpec(CommandLineInputSpec):
#class MRtrix3DenoiseInputSpec(ANTSCommandInputSpec):
class dwidenoiseInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, position=0, argstr="%s")
    out_file = File(
        argstr="%s",
        name_source=['in_file'],  # use with self._filename_from_source, failed for MultiProc
        hash_files=False,
        genfile=True,
        position=1,
        keep_extension=True,
        name_template='%s_denoised',  # use with self._filename_from_source, failed for MultiProc
        desc='The output consists of the denoised '
             'version of the input image.')

class dwidenoiseOutputSpec(TraitedSpec):
    out_file = File(exists=True)

#class MRtrix3Denoise(CommandLine):
class dwidenoise(FSLCommand):
#class MRtrix3Denoise(ANTSCommand):
    """
    Borrow from N4BiasFieldCorrection
    """
    input_spec  = dwidenoiseInputSpec
    output_spec = dwidenoiseOutputSpec
    _cmd = 'dwidenoise'

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_fname(self.inputs.in_file, suffix="_reoriented")
        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        #outputs = self.output_spec().get()
        #outputs['out_file'] = os.path.abspath(
        #    self._gen_filename('out_file'))

        #Below work only for MultiProc, failed for SGE, SGEGraph both
        outputs['out_file'] = os.path.abspath(self._filename_from_source('out_file'))
        return outputs
