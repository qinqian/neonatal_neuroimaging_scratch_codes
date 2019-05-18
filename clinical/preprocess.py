#!/usr/bin/env python

import os
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.algorithms.rapidart import ArtifactDetect
from nipype.interfaces.utility import IdentityInterface, Function

def parse_dcmdir(in_file):
    import dicom
    try:
        z=dicom.read_file(in_file)
        patient_id=str(z.DirectoryRecordSequence[0][('0010', '0020')]).split(":")[1].strip("' ")
        accession_id=str(z.DirectoryRecordSequence[1][('0008', '0050')]).split(":")[1].strip("' ")
        date=str(z.DirectoryRecordSequence[1][('0008', '0020')]).split(":")[1].strip("' ")
        time=str(z.DirectoryRecordSequence[1][('0008', '0030')]).split(":")[1].strip("' ")
        return in_file, accession_id, patient_id, date, time
    except:
        return in_file, 'error', 'error', 'error','error'

def list_all_dicomdir():
    f = os.popen('find ../data/MRI/fMRI ../data/MRI/gene/ ../data/MRI/MRI/ ../data/MRI/other_gene/ -iname "dicomdir"')
    # print("%s\t%s\t%s\t%s\t%s" % ('directory', 'accession_id', 'patient_id', 'date', 'time'))
    for line in f:
        line = line.strip()
        yield line
        # print('\t'.join(parse_dcmdir(line)))


def create_step1(dicoms):
    c = Dcm2niix()
    step1_dcm2nii = MapNode(c, name="dcm2niix", iterfield=['source_dir'])
    step1_dcm2nii.inputs.source_dir = dicoms
    step1_dcm2nii.inputs.compress = 'i'
    step1_dcm2nii.inputs.single_file = False
    step1_dcm2nii.inputs.out_filename = '%s_%i_%t_%p_%v'
    step1_dcm2nii.inputs.output_dir = "."
    return step1_dcm2nii

def create_identity(l):
    infosource = Node(IdentityInterface(fields=['subject_id']), name="info source")
    infosource.iterables = [('subject_id', l)]

dicomdir_nd = MapNode(Function(input_names = ['in_file'],
                               output_names = ('directory', 'accession_id', 'patient_id', 'date', 'time'),
                               function = parse_dcmdir),
                      name="dicom_meta", iterfield=['in_file'])
dicomdir_nd.inputs.in_file = map(os.path.abspath, list(list_all_dicomdir()))

# ----------- whole automatic pipeline ----------------
experiment_dir = '../data/'
work_dir = 'workdir'
#work_dir = 'workdir_test'
pre = Workflow(name="preproc")
pre.base_dir = os.path.join(experiment_dir, work_dir)

n = 0
dicom_all = []
with open('MRI_batch1.list.sorted') as fin:
    for line in fin:
        line = line.strip().split()
        folder = os.path.dirname(line[0])
        if 'fMRI' in folder:
            for f in os.listdir(folder):
                if f.lower() == 'dicom' or f.startswith('SDY'):
                    n += 1
                    dicom_all.append(os.path.join(folder, f))

step1_node = create_step1(dicom_all)

pre.add_nodes([dicomdir_nd, step1_node])
#pre.write_graph(graph2use='flat')
#pre.run(plugin='MultiProc', plugin_args={'n_procs': 4})
pre.run(plugin='SGE', plugin_args={'qsub_args': ' -cwd -q normal.q '})

