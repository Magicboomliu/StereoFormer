swin_former_cls(){
cd ..
cd train_file

data_path=/media/zliu/datagrid1/liu/flower_photos
type=classification
in_channels=3
patch_size=4
window_size=7
embedding_dim=96
num_classes=1000
epochs=100
batch_size=4
lr=1e-4
weights=/home/zliu/Desktop/Codes/StereoFormer/pretrained/swin_tiny_patch4_window7_224.pth



python -W ignore classification.py --type $type \
                                    --in_channels $in_channels \
                                    --patch_size $patch_size \
                                    --window_size $window_size \
                                    --num_classes $num_classes \
                                    --embedding_dim $embedding_dim \
                                    --epochs $epochs \
                                    --batch_size $batch_size \
                                    --lr $lr \
                                    --weights $weights \
                                    --data_path $data_path

}

swin_former_cls