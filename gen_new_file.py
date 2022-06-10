import os

from setuptools import find_packages

from utils.utils import read_text_lines



def GetFileListByOrder(file_path):
    filelist = os.listdir(file_path)
    filelist = sorted(filelist, key=lambda y: int(y[0:7]))
    
    filelist_absolute =[os.path.join(file_path,p) for p in filelist]
    
    return filelist_absolute

def Existence_Check(file_name_list):
    for f in file_name_list:
        if not os.path.exists(f):
            return False
    else:
        return True

def GenFileByOrderWithCheck(root_path,file_path):
    filelist_absolute = GetFileListByOrder(file_path)
    ret = Existence_Check(filelist_absolute)
    if ret:
        filelist_absolute = [f[len(root_path)+1:] for f in filelist_absolute]
        return filelist_absolute
    else:
        raise NotImplementedError



if __name__=="__main__":
    
    sceneflow_root_file = "/media/zliu/datagrid1/liu/sceneflow"
    
    training_filename ='filenames/SceneFlow.list'
    
    flythings3D_subset ="frames_cleanpass/flythings3d/"
        
    lines = read_text_lines(training_filename)
    
    cnt =  0
    for idx, instances in enumerate(lines):
        if instances[:len(flythings3D_subset)] == flythings3D_subset:
            cnt = cnt +1
    # 22390
    print(cnt)
    
    

    # occlusion_mask_border_mask_root_path_root =  os.path.join(sceneflow_root_file,"FlyingThings3D_subset")
    
    # # Train Path
    # occlusion_mask_root_path_train = os.path.join(occlusion_mask_border_mask_root_path_root,"train/depth_boundaries")
    # occlusion_mask_root_path_train_left = os.path.join(occlusion_mask_root_path_train,'left')
    # occlusion_mask_root_path_train_right = os.path.join(occlusion_mask_root_path_train,'right')
    # border_mask_root_path_train = os.path.join(occlusion_mask_border_mask_root_path_root,"train/disparity_occlusions")
    # border_mask_root_path_train_left = os.path.join(border_mask_root_path_train,'left')
    # border_mask_root_path_train_right = os.path.join(border_mask_root_path_train,'right')
    
    # #Validation Path
    # occlusion_mask_root_path_val = os.path.join(occlusion_mask_border_mask_root_path_root,"val/depth_boundaries")
    # occlusion_mask_root_path_val_left = os.path.join(occlusion_mask_root_path_val,'left')
    # occlusion_mask_root_path_val_right = os.path.join(occlusion_mask_root_path_val,'right')
    # border_mask_root_path_val = os.path.join(occlusion_mask_border_mask_root_path_root,"val/disparity_occlusions")
    # border_mask_root_path_val_left = os.path.join(border_mask_root_path_val,'left')
    # border_mask_root_path_val_right = os.path.join(border_mask_root_path_val,'right')
    
    
    # border_list_files_train = GenFileByOrderWithCheck(sceneflow_root_file,border_mask_root_path_train_left)
    
    # print(len(border_list_files_train))

    
    
    
    
    
    
    
    
    


