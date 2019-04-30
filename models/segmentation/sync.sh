
#scp augmentations/ 2d_pvls *sh  *py *csv *xls *txt qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg
#scp -r models  *sh  *py  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg
#scp -r 2d_pvls  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg
#scp -r losses.py  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg
#scp -r keras_deeplabv3plus.py losses.py train_unet.py  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg
#scp -r qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/losses.py .
#scp -r qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/train_unet.py .
#scp -r qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/prediction_valid.py .
#scp -r qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/run.sh .
#scp -r sync.sh train_unet.py  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/
#scp -r sync.sh data_3channel_with_mask.xls  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/
#scp -r test.sh  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/
#scp -r sync.sh test_dense_ok.sh unet.py train_unet.py prediction_valid.py qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/
#scp run.sh sync.sh prediction_valid.py qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/
#scp -r  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/test_dense_ok.sh .
#scp -r  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/test_dense_ok.sh .
#scp -r mask qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg

#scp train_unet.py qinqian@193.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/
#scp train_unet.py stat_epochs_dice_dr.py stat_lesion.py qinqian@193.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/
scp train_unet.py  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/
#scp  qinqian@192.168.165.1:/cluster/home/qinqian/phaseC_pvl_seg/generate_patches_tfrecords.py .

