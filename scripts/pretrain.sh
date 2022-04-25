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

HRNet_GWc_Sf