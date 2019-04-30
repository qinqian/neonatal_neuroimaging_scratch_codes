#!/usr/bin/env python

""" 
input dicomdir
output a dict key(modal name)->value(dicom directory)
"""

import sys, os
import argparse

def get_property(x, p):
    if hasattr(x[0], p):
        y = x[0].__getattr__(p)
    elif hasattr(x[1], p):
        y = x[1].__getattr__(p)
    return y

def parse_dcmdir(in_file):
    import dicom
    z=dicom.read_file(in_file)
    patient_id=get_property(z.DirectoryRecordSequence, 'PatientID')
    birth=get_property(z.DirectoryRecordSequence, 'PatientBirthDate')
    sex=get_property(z.DirectoryRecordSequence, 'PatientSex')

    dicom_dict = {}
    for i in z.DirectoryRecordSequence[1:]:
        if hasattr(i, 'SeriesDescription'):
            key = i.SeriesDescription
            dicom_dict[key] = {}
            if hasattr(i, 'SeriesDate'):
                date = i.SeriesDate
            else:
                date = 'NA'
            if hasattr(i, 'SeriesTime'):
                time = i.SeriesTime
            else:
                time = 'NA'
            if hasattr(i, 'SeriesNumber'):
                sn = i.SeriesNumber
            else:
                sn = 'NA'

            dicom_dict[key][sn] = {'images': [], 'info': []}
            dicom_dict[key][sn]['info'] = [date, time, 0]
        elif i.DirectoryRecordType == "IMAGE":
            dicom_dict[key][sn]['images'].append(os.path.join(os.path.dirname(in_file), os.path.join(*i.ReferencedFileID)))
            dicom_dict[key][sn]['info'][-1] += 1

    for k in dicom_dict: # each modal
        for s in dicom_dict[k]: # each series
            print '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (patient_id, sex, birth, k, s, '\t'.join(map(str, dicom_dict[k][s]['info'])), in_file, ','.join(dicom_dict[k][s]['images']))

    return in_file, patient_id, birth

if __name__ == '__main__':
    p = argparse.ArgumentParser('parse dicomdir to multi-modal dicom directory')
    p.add_argument('-d', help='dicom dir')

    args = p.parse_args()

    if not args.d:
        p.print_help()
        sys.exit(1)

    parse_dcmdir(args.d)

