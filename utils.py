from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import os

INPUT_SIZE = 608

def parse_config(config_file):
    " Read the official yolo config file correctly"
    with open(config_file) as f:
        lines = f.readlines()
    config = []
    for line in lines:
        if line and line != '\n' and line[0] != '#':
            stripped_line = line.replace(' ', '').replace('\n', '')
            if stripped_line[0] == '[':
                try:
                    config.append(data)
                except UnboundLocalError:
                    pass
                data = {}
                data['type'] = stripped_line[1:-1]
            else:
                [key, value] = stripped_line.split('=')
                data[key] = value
    config.append(data)
    network_info = config[0]
    config.pop(0)
    return config, network_info

def get_all_bboxes(prediction, inp_dim, anchors, num_classes, use_gpu=False):
    "takes feature map and returns all bboxes as rows"
    # data size: batch_size x ? x grid x grid
    batch_size, _, grid_size, _ = prediction.size()
    stride =  inp_dim // prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    # batch_size x all rows x vector length
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if use_gpu:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if use_gpu:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction

def load_test_image(path=os.path.join('data', 'dog-cycle-car.png')):
    img = cv2.imread(path)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W check this
    img_ = img_[np.newaxis,:,:,:] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_