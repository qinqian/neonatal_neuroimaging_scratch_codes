#!/bin/bash -ex

mkdir -p results/processed
mkdir -p results/template_candidates
mkdir -p results/raw
mkdir -p results/class_models

preprocess() {
   # for i in `cut -f 1 -d, results/bids_head1.csv`; do
   #     #qsub preprocess.pbs -F "$i"
   #     echo $i $(fslval results/processed/${i}_adc_processed.nii.gz dim1) $(fslval results/processed/${i}_adc_processed.nii.gz dim2) $(fslval results/processed/${i}_adc_processed.nii.gz dim3)
   # done

   #visualize by slicesdir and throw out datasets
   #mkdir -p results/ants_processed_bad
   #for i in `cut -f 1 -d" " check_datasets/bad_ids_thrown_ids_for_antspy.txt`; do
   #    mv results/processed/${i}_* results/ants_processed_bad  
   #done

   #for i in `cut -f 1 -d" " check_datasets/bad_ids_thrown_ids_for_antspy.txt`; do
   #    #adc=$(grep "^${i}," results/bids_head1.csv | cut -d, -f 11)
   #    #echo $i $(fslval $adc dim1) $(fslval $adc dim2) $(fslval $adc dim3)
   #    t2=$(grep "^${i}," results/bids_head1.csv | cut -d, -f 7)
   #    echo $i $(fslval $t2 dim1) $(fslval $t2 dim2) $(fslval $t2 dim3)
   #done

   #for i in `cut -f 1 -d" " check_datasets/bad_ids_thrown_ids_for_antspy2.txt`; do
   #    mv results/processed_fsl_bad/${i}_* results/ants_processed_bad  
   #done

   #while read line; do
   #    echo $line
   #    python preprocess_with_casetemplate.py --bids results/bids_head1.csv --inputid $line
   #done < <(grep "113 9" check_dims.txt | cut -f 1 -d" ")

   while read i; do
       echo $i $(fslval results/processed/${i}_adc_processed.nii.gz dim1) $(fslval results/processed/${i}_adc_processed.nii.gz dim2) $(fslval results/processed/${i}_adc_processed.nii.gz dim3)
   done < <(grep "113 9" check_dims.txt | cut -f 1 -d" ")
}

main() {
   #preprocess

   # search for optimal learning rate, each label is different Leukomalacia 0.00017
   # mkdir -p results/learning_rate/
   # python main.py -modelpath results/learning_rate/ -endlr 0.0005 -findlr -gpu 2 -symptom Leukomalacia -drop 0 -lr 0.00005 -batch 16 -epoch 100 -bids results/bids_head1.csv -exclude check_datasets/bad_ids_thrown_ids_for_antspy2.txt -modality 0,1,2,3,4

   # optimal learning rate lr_test.pdf 0.0001625 
   # python main.py -modelpath results/learning_rate/ -endlr 0.000175 -findlr -gpu 0 -symptom Hemorrhage -drop 0 -lr 0.00015 -batch 32 -epoch 100 -bids results/bids_head1.csv -exclude check_datasets/bad_ids_thrown_ids_for_antspy2.txt -modality 0,1,2,3,4

   rm -f *_more_combination_class_training_valid_gpu*
   gpu_num=0
   #for symptom in Leukomalacia Hemorrhage Expansion Lesion Cyst Myelin_or_Develop; do
   for symptom in Leukomalacia; do
       # for modality
       if [ $symptom == 'Leukomalacia' ]; then
           lr=0.00017
       else
           lr=0.0001
       fi
       for modality in 0 "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" 1 2 3 4 0,2 0,3 0,4 1,2 1,3 1,4 2,3 2,4 3,4 0,1,3 0,1,4 0,2,3 0,2,4 0,3,4 1,2,3 1,2,4 1,3,4 2,3,4 0,1,2,4 0,1,3,4 0,2,3,4 1,2,3,4; do
           #parameter set
           for drops in 0.5 0.2; do # regularization
               for batch in 32; do
                   if [ $gpu_num -ge 3 ]; then
                       gpu_num=0
                   fi
                   #if [ ! -s results/${symptom}_modal${modality}_class_models_initiallr${lr}_batch${batch}_drop${drops}.logs ]; then
                   echo "python main.py -modelpath results/models_${symptom}_pure -gpu $gpu_num -symptom $symptom -drop $drops -lr $lr -batch $batch -epoch 100 -bids results/bids_head1.csv -exclude check_datasets/bad_ids_thrown_ids_for_antspy2.txt -modality $modality 2>&1>results/${symptom}_modal${modality}_class_models_initiallr${lr}_batch${batch}_drop${drops}.logs" >> ${symptom}_more_combination_class_training_valid_gpu${gpu_num}.commands
                   #fi
                   let gpu_num++
               done
           done
       done
       rm *_more_top5_*
       for command in ${symptom}_more_combination_class_training_valid_gpu*.commands; do
           sed -n '1,8p' $command > ${symptom}_more_top5_${command}
       done
   done

   #bash Leukomalacia_more_top5_Leukomalacia_more_combination_class_training_valid_gpu0.commands
   #bash Leukomalacia_more_top5_Leukomalacia_more_combination_class_training_valid_gpu1.commands
   #bash Leukomalacia_more_top5_Leukomalacia_more_combination_class_training_valid_gpu2.commands

   #mkdir -p results/models_Leukomalacia_head1
   #for i in `cut -f 1 results/models_head1.txt`; do
   #    mkdir -p results/models_Leukomalacia_head1/$(basename $(dirname $i))
   #    scp qinqian@192.168.165.1:$(dirname $i)*csv results/models_Leukomalacia_head1/
   #    scp qinqian@192.168.165.1:${i} results/models_Leukomalacia_head1/$(basename $(dirname $i))
   #done

   # test and visualize by jupyter notebook
   python main.py -resume -modelpath results/models_Leukomalacia_seed1_pure -gpu 0 -symptom Leukomalacia -drop 0.5 -lr 0.00017 -batch 32 -epoch 100 -bids results/bids_head1.csv -exclude check_datasets/bad_ids_thrown_ids_for_antspy2.txt -modality 0
}

main

