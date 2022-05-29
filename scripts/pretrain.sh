HRNet_GWc_Sf(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/HRNet_GWc
logf=logs/HRNet_GWc
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow.list
vallist=filenames/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=1
testbatch=1
maxdisp=-1
model=none
save_logdir=experiments_logdir/HRNet_GWc
model=Swin_t
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

Swin_T_StereoNet_Sf(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/Swin_stereonet
logf=logs/Swin_stereonet
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow.list
vallist=filenames/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=8
testbatch=4
maxdisp=-1
model=none
save_logdir=experiments_logdir/Swin_stereoNet
model=Swin_t
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

HRNet_StereoNet_Sf(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/hrnet_stereonet
logf=logs/hrnet_stereonet
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
testbatch=1
maxdisp=-1
model=none
save_logdir=experiments_logdir/Hrnet_stereoNet
model=HRNet
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

Transs(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/ninet_trans
logf=logs/ninet_trans
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow.list
vallist=filenames/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=8
testbatch=8
maxdisp=-1
model=none
save_logdir=experiments_logdir/ninet_trans
model=NiNet
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



TransUnet_Low_Scale(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/transunet_low_scale_simple_correlation
logf=logs/transunet_low_scale_simple_correlation
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow.list
vallist=filenames/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=8
testbatch=1
maxdisp=-1
model=none
save_logdir=experiments_logdir/transunet_low_scale_simple_correlation
model=TransUnet
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
batchSize=8
testbatch=8
maxdisp=-1
model=none
save_logdir=experiments_logdir/LowCNN
model=LowCNN
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

Raft_stereo(){
cd ..
loss=config/loss_config_disp.json
outf_model=models_saved/ninet_trans
logf=logs/ninet_trans
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=filenames/SceneFlow.list
vallist=filenames/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=1
testbatch=1
maxdisp=-1
model=none
save_logdir=experiments_logdir/ninet_trans
model=RAFT
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








Raft_stereo

# HRNet_StereoNet_Sf
#Swin_T_StereoNet_Sf
# HRNet_GWc_Sf
# LowCNN
# Transs
# TransUnet_Low_Scale