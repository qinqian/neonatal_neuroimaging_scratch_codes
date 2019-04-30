#!/bin/bash
#$ -V
#$ -S /bin/bash
export PYTHONPATH=/data1/qinqian/MRI/hyperbilirubinemia/scripts:${PYTHONPATH}

FSLDIR=/home/qinqian/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
