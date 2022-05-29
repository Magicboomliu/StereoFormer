import os
import sys




sceneflow_saved_path = "/media/zliu/datagrid1/liu/sceneflow"

frame_cleanpass = 'frames_cleanpass'
disparity = 'disparity'

flyingthings3d_source = 'flythings3d'
driving_source = 'driving'
monkaa_source = 'monkaa'

flythings3d_target = "FlyingThings3D"
monkaa_target = "Monkaa"
driving_target = "Driving"

os.makedirs("datasets")
# flythings3D
FT_root = "datasets/{}".format(flythings3d_target)
os.makedirs(FT_root)

os.system("ln -s {}/frames_cleanpass/flythings3d/ {}/frames_cleanpass".format(sceneflow_saved_path,FT_root))
os.system("ln -s {}/disparity/flythings3d/ {}/disparity".format(sceneflow_saved_path,FT_root))
os.system("ln -s {}/frames_cleanpass/flythings3d/ {}/frames_finalpass".format(sceneflow_saved_path,FT_root))

# monkaa
MK_root = "datasets/{}".format(monkaa_target)
os.makedirs(MK_root)

os.system("ln -s {}/frames_cleanpass/monkaa/ {}/frames_cleanpass".format(sceneflow_saved_path,MK_root))
os.system("ln -s {}/disparity/monkaa/ {}/disparity".format(sceneflow_saved_path,MK_root))
os.system("ln -s {}/frames_cleanpass/monkaa/ {}/frames_finalpass".format(sceneflow_saved_path,MK_root))

# driving
DR_root = "datasets/{}".format(driving_target)
os.makedirs(DR_root)

os.system("ln -s {}/frames_cleanpass/driving/ {}/frames_cleanpass".format(sceneflow_saved_path,DR_root))
os.system("ln -s {}/disparity/driving/ {}/disparity".format(sceneflow_saved_path,DR_root))
os.system("ln -s {}/frames_cleanpass/driving/ {}/frames_finalpass".format(sceneflow_saved_path,DR_root))


