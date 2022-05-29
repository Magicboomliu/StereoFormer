import imp
import os
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.common import count_parameters 
from dataloader import transforms
from dataloader.SceneflowLoader import StereoDataset
import matplotlib.pyplot as plt
import numpy as np
import time

from utils.devtools import disp_error_img
from dataloader.preprocess import scale_disp

# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]# ImageNet Normalization

from models.TwoD.nitnet_pp import NiNet

from models.TwoD.pureCNN import LowCNN

pretrained_path = "/home/zliu/Desktop/Codes/StereoFormer/pretrained/backbone/upernet_swin_tiny_patch4_window7_512x512.pth"

def load_model(model_path,type='nitnet'):
    if type =='nitnet':
        net = NiNet(res_type='context_attention',squeezed_volume=True,load_swin_pretrain=True,
                  swin_transformer_path=pretrained_path,fixed_parameters=True)
    elif type =='lowcnn_trans':
        net = LowCNN(cost_volume_type='correlation',upsample_type='convex')
    elif type =='lowcnn_cnn':
        net = LowCNN(cost_volume_type='correlation',upsample_type='convex')
    else:
        raise NotImplementedError
    
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    
    model_data = torch.load(model_path)
    print(model_data.keys())

    if 'state_dict' in model_data.keys():
        net.load_state_dict(model_data['state_dict'])
    else:
        net.load_state_dict(model_data)
    num_of_parameters = count_parameters(net)
    print('Model: %s, # of parameters: %d' % ("Disp", num_of_parameters))
    return net
    



def prepare_dataset(file_path,train_list,val_list):
    test_batch =1
    num_works = 1
    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
    val_transform = transforms.Compose(val_transform_list)

    test_dataset = StereoDataset(data_dir=file_path,train_datalist=train_list,test_datalist=val_list,
                                    dataset_name='SceneFlow',mode='val',transform=val_transform)
    scale_height, scale_width = test_dataset.get_scale_size()
    
    test_loader = DataLoader(test_dataset, batch_size = test_batch, \
                                shuffle = False, num_workers = num_works, \
                                pin_memory = True)
    return test_loader



if __name__=="__main__":

    file_path = "/media/zliu/datagrid1/liu/sceneflow"
    train_list = "filenames/SceneFlow.list"
    val_list = "filenames/FlyingThings3D_release_TEST.list"
    saved_dir = "results/low_cnn"
    model_path = "/home/zliu/Desktop/Codes/Models/cost_volume/cnncost.pth"
    
    
    disp_gt_path = os.path.join(saved_dir,"disp_gt")
    disp_infer_path = os.path.join(saved_dir,"disp_infer")
    disp_error_path = os.path.join(saved_dir,"disp_error")
    
    left_path = os.path.join(saved_dir,"left")
    right_path = os.path.join(saved_dir,"right")
    
    if os.path.exists(saved_dir):
        pass
    else:
        os.makedirs(saved_dir)
        os.makedirs(disp_gt_path)
        os.makedirs(disp_infer_path)
        os.makedirs(disp_error_path)
        os.makedirs(left_path)
        os.makedirs(right_path)
        
        
    test_loader = prepare_dataset(file_path=file_path,train_list=train_list,val_list=val_list)
    print("***DATA LOADED")
    
    pretrained_model = load_model(model_path=model_path,type='lowcnn_cnn')
    
    pretrained_model.eval()
    print("Model Loaded")
    


    for i, sample_batched in enumerate(test_loader):
        left_input = Variable(sample_batched['img_left'].cuda(), requires_grad=False)
        right_input = Variable(sample_batched['img_right'].cuda(), requires_grad=False)

        target_disp = sample_batched['gt_disp']
        target_disp = target_disp.cuda()
        target_disp =Variable(target_disp, requires_grad=False)
        
        # GT disparity Vis
        target_disp_vis = target_disp.squeeze(0).squeeze(0).cpu().numpy()
        
        with torch.no_grad():
            inference_time =0
            start_time = time.perf_counter()
            disparity= pretrained_model(left_input,right_input,False)
            inference_time += time.perf_counter() - start_time
            disparity = scale_disp(disparity, (disparity.size()[0], 540, 960))

            infer_disparity = disparity.squeeze(0).squeeze(0).cpu().numpy()

            disp_errors = disp_error_img(disparity.squeeze(1),target_disp.squeeze(1))
            disp_error_vis = disp_errors.squeeze(0).permute(1,2,0).cpu().numpy()
            
            
            left_rgb = left_input.squeeze(0).permute(1,2,0).cpu().numpy()
            
            left_rgb = left_rgb * IMAGENET_STD + IMAGENET_MEAN
            
            right_rgb = right_input.squeeze(0).permute(1,2,0).cpu().numpy()
            
            right_rgb = right_rgb * IMAGENET_STD + IMAGENET_MEAN
            
            
            plt.figure(figsize=(10,8))
            plt.axis("off")
            plt.imshow(target_disp_vis)
            plt.savefig("{}/gt_disp_{}.png".format(disp_gt_path,i),bbox_inches = 'tight',pad_inches = 0)
            
            plt.figure(figsize=(10,8))
            plt.axis("off")
            plt.imshow(infer_disparity)
            plt.savefig("{}/disp_infer_{}.png".format(disp_infer_path,i),bbox_inches = 'tight',pad_inches = 0)

            plt.figure(figsize=(10,8))
            plt.axis("off")
            plt.imshow(disp_error_vis)
            plt.savefig("{}/disp_vis_{}.png".format(disp_error_path,i),bbox_inches = 'tight',pad_inches = 0)
            

            plt.figure(figsize=(10,8))
            plt.axis("off")
            plt.imshow(left_rgb)
            plt.savefig("{}/left_{}.png".format(left_path,i),bbox_inches = 'tight',pad_inches = 0)

            plt.figure(figsize=(10,8))
            plt.axis("off")
            plt.imshow(right_rgb)
            plt.savefig("{}/right_{}.png".format(right_path,i),bbox_inches = 'tight',pad_inches = 0)
            
            pass
            
            
            
            
    