import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo):
    #t1w = create_key('sub-{subject}/anat/sub-{subject}_run-{item:01d}_T1w')
    #dwi = create_key('sub-{subject}/dwi/sub-{subject}_run-{item:01d}_dti')

    t1w = create_key('sub-{subject}/anat/sub-{subject}_T1w')
    #t2w = create_key('sub-{subject}/anat/sub-{subject}_T2w') # too few slices
    dwi = create_key('sub-{subject}/dwi/sub-{subject}_run-{item:01d}_dti')
    dki = create_key('sub-{subject}/dwi/sub-{subject}_run-{item:01d}_dki')
    bold = create_key('sub-{subject}/func/sub-{subject}_run-{item:01d}_bold')
    asl = create_key('sub-{subject}/func/sub-{subject}_run-{item:01d}_asl')

    info = {t1w: [], dwi: [],
            dki: [], bold: [], asl: []}
    for s in seqinfo:
        #if 'T2' in s.series_description:
        #    info[t2w].append(s.series_id) 
        if 'BRAVO' in s.series_description:
            info[t1w].append(s.series_id) 
        if 'DTI' in s.series_description:
            info[dwi].append(s.series_id)

        if 'BOLD' in s.series_description:
            info[bold].append(s.series_id)

        if 'DKI' in s.series_description:
            info[dki].append(s.series_id)
        if 'ASL' in s.series_description:
            info[asl].append(s.series_id)
    return info
