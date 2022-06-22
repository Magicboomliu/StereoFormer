import os
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
import matplotlib.pyplot as plt
import numpy as np
# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]# ImageNet Normalization

from models.LocalCostVolume.baseline import LowCNN
from models.utils.estimation import DisparityEstimationWithProb
from models.LocalCostVolume.Attempts.fixed_local_cost_volume import LocalCostVolume

disp_estimation3 = DisparityEstimationWithProb(max_disp=24,match_similarity=True) 

def load_model(model_path,type='simple_cost_volume'):
    if type =='simple_cost_volume':
        net = LowCNN(cost_volume_type='correlation',upsample_type='convex',adaptive_refinement=False)
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
    # saved_dir = "results/low_cnn"
    model_path = "/home/zliu/Desktop/Codes/Models/cost_volume/mostsimpleCNN.pth"
    
    test_loader = prepare_dataset(file_path=file_path,train_list=train_list,val_list=val_list)
    print("***DATA LOADED")
    
    pretrained_model = load_model(model_path=model_path,type='simple_cost_volume')
    
    pretrained_model.eval()
    print("Model Loaded")
    
    for i, sample_batched in enumerate(test_loader):
        left_input = Variable(sample_batched['img_left'].cuda(), requires_grad=False)
        right_input = Variable(sample_batched['img_right'].cuda(), requires_grad=False)

        target_disp = sample_batched['gt_disp']
        target_disp = target_disp.cuda()
        target_disp =Variable(target_disp, requires_grad=False)
        
        fines = scale_disp(target_disp.unsqueeze(1),(target_disp.unsqueeze(1).size()[0], 576, 960))
        target_disp_low = F.interpolate(fines,scale_factor=1./8,mode='bilinear',align_corners=False)/8
        
        # GT disparity Vis
        target_disp_vis = target_disp.squeeze(0).squeeze(0).cpu().numpy()
        
        new_cost_volume = LocalCostVolume(radius=2.1,sample_points=20).cuda()
        
        with torch.no_grad():
            
            inference_time =0
            start_time = time.perf_counter()
            disparity, low_disp3,final_cost= pretrained_model(left_input,right_input,False)
            
            infer_disparity= disparity.squeeze(0).squeeze(0).cpu().numpy()
            prob_volume = np.load("/home/zliu/Desktop/Codes/StereoFormer/prob_volume.npy")
            prob_volume_tensor = torch.Tensor(prob_volume).permute(2,0,1).unsqueeze(0).type_as(low_disp3)

            
            new_disp = new_cost_volume(prob_volume_tensor,low_disp3,True)
            
            
            xy_coordinate = [39,21]        
            prob_list = prob_volume[xy_coordinate[0],xy_coordinate[1],:]
            pred_disp_new = new_disp[0,:,xy_coordinate[0],xy_coordinate[1]].cpu().item()
            pred_disp = low_disp3[0,:,xy_coordinate[0],xy_coordinate[1]].cpu().item()
            gt_disp = target_disp_low[0,:,xy_coordinate[0],xy_coordinate[1]].cpu().item()
            plt.title("old/new pred Disp is: {}/{} GT is {} ".format(pred_disp,pred_disp_new,gt_disp))
            plt.xlabel("disparity searching range")
            plt.ylabel("Probality")
            plt.plot(prob_list)
            plt.axvline(pred_disp,c='r')
            plt.axvline(gt_disp,c='green')
            # plt.axvline(new_pred_disp_new,c='blue')
            plt.show()
        
            break
    
    



