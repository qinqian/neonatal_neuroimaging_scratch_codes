#!/bin/bash -ex

merge_niis() {
    mkdir -p results
    mkdir -p tfrecords
    rm -f preprocess_R.commands
    while read line;do
        field=($line)
        echo Rscript preprocess.R ${field[1]} ${field[2]} ${field[3]} ${field[4]} ${field[5]} ${field[0]}  >> preprocess_R.commands
    #done < <(cut -f 2,4-8 results/data_for_model.xls | sed 1d)
    done < <(cut -f 2,4-8 results/data_for_model_manual_fixed_resample_failure.xls | sed 1d)
    parallel -j 28 < preprocess_R.commands
}

write_tfrecord() {
    #rm -f tfrecord.commands
    #for i in nii/*__not_iso_merge_chan.nii.gz; do
    #    echo $i
    #    echo python generate_tfrecord.py --infile $i >> tfrecord.commands
    #done
    #parallel -j 16 < tfrecord.commands

    #rm -f tfrecord.commands.lower
    #for i in nii/*lower*nii.gz; do
    #    echo $i
    #    echo python generate_tfrecord.py --infile $i >> tfrecord.commands.lower
    #done
    #parallel -j 16 < tfrecord.commands.lower

    rm -f tfrecord.commands.uncnorm
    for i in norm/*nii.gz; do
        echo $i
        echo python generate_tfrecord.py --infile $i >> tfrecord.commands.uncnorm
    done
    parallel -j 42 < tfrecord.commands.uncnorm

}

main() {
   #qsub -q q1 -t 4-1900 norm.pbs # job array weird
   write_tfrecord
}

main
