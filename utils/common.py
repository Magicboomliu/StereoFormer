import json
import yaml
import logging
import os
import numpy as np
import sys

def load_loss_scheme(loss_config):

    with open(loss_config, 'r') as f:
        loss_json = yaml.safe_load(f)

    return loss_json

DEBUG =0
logger = logging.getLogger()

if DEBUG:
    #coloredlogs.install(level='DEBUG')
    logger.setLevel(logging.DEBUG)
else:
    #coloredlogs.install(level='INFO')
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)


#from netdef_slim.utils.io import read 
#left_img = sys.argv[1]
#subfolder = sys.argv[2]
#
#occ_file = 'tmp/disp.L.float3'
#occ_data = read(occ_file) # returns a numpy array
#
#import matplotlib.pyplot as plt
#occ_data = occ_data[::-1, :, :] * -1.0
#print(np.mean(occ_data))
##plt.imshow(occ_data[:,:,0], cmap='gray')
## plt.show()
#
#subfolder = "detect_results/%s" % subfolder
#if not os.path.exists(subfolder):
#    os.makedirs(subfolder)
#
##name_items = left_img.split('.')[0].split('/')
##save_name = '_'.join(name_items) + '.pfm'
#name_items = left_img.split('/')
#filename = name_items[-1]
#topfolder = name_items[-2]
#save_name = filename + '.pfm'
#target_folder = '%s/%s' % (subfolder, topfolder)
#print('target_folder: ', target_folder)
#if not os.path.exists(target_folder):
#    os.makedirs(target_folder)
#save_pfm('%s/%s' % (target_folder, save_name), occ_data[:,:,0])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)