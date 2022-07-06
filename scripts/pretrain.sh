

LowCNN(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/lowCNN
logf=logs/LowCNN
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow.list
vallist=filenames/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=6
testbatch=8
maxdisp=-1
model=none
save_logdir=experiments_logdir/LowCNN
model=SwinOnly
pretrain=none

python3 -W ignore train_low.py --cuda --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --pretrain $pretrain  
}

CMAStereo(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/CMAStereo
logf=logs/CMAStereo
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow.list
vallist=filenames/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=4
testbatch=8
maxdisp=-1
model=none
save_logdir=experiments_logdir/LowCNN
model=CMAStereo
pretrain=none

python3 -W ignore train_iter.py --cuda --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --pretrain $pretrain  
}

RAFTStereo(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/RAFTStereo
logf=logs/RAFTStereo
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow.list
vallist=filenames/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=2
testbatch=8
maxdisp=-1
model=none
save_logdir=experiments_logdir/RAFTStereo
model=RAFT
pretrain=none

python3 -W ignore train.py --cuda --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --pretrain $pretrain  
}

RAFTStereo
# CMAStereo
# LowCNN
# Raft_stereo
# LowCNN_test2
# HRNet_StereoNet_Sf
#Swin_T_StereoNet_Sf
# HRNet_GWc_Sf
# LowCNN
# Transs
# TransUnet_Low_Scale