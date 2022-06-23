
LowCNN(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/lowCNN_localCosTVolume_Variance_samples20
logf=logs/lowCNN_localCosTVolume_Variance_samples20
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
save_logdir=experiments_logdir/lowCNN_localCosTVolume_Variance_samples20
model=LowCNN_ada
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



Test(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/Fuck
logf=logs/Fuck
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
save_logdir=experiments_logdir/Fuck
model=Baseline_ca
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


# LowCNN
Test
