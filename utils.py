from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import os
import torchvision.ops.boxes as bops

INPUT_SIZE = 608

def resize_with_ratio(img):
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(INPUT_SIZE)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = INPUT_SIZE - new_size[1]
    delta_h = INPUT_SIZE - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


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
    # ? is number of attributes in prediction vector x n_anchors per grid box
    batch_size, _, grid_size, _ = prediction.size()
    stride =  inp_dim // prediction.size(2) # 32, 16 or 8
    bbox_attrs = 5 + num_classes # TODO: move this as global, but maybe check if classes don't change
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    # batch_size x all rows x vector length
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # transformation of objectivness
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    # transformation of x,y coordinates
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    if use_gpu:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,:2] += x_y_offset

    # transformation of h wand w
    anchors = torch.FloatTensor(anchors)
    if use_gpu:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = anchors * torch.exp(prediction[:,:,2:4])
    
    # transformation of class prediction
    prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])

    # upscaling from feature map to original image size
    # TODO: need to check whether the transformation happens before or after upscaling
    prediction[:,:,:4] *= stride
    
    return prediction

def load_test_image(path=os.path.join('data', 'debugging_data', 'dog-cycle-car.png')):
    img = cv2.imread(path)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def nms(dets, thresh):
    # check this
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = dets[:, 4].sort(descending=True)[1]
    keep = []
    while order.size()[0] > 0:
        i = order[0].item()
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def get_results(prediction, min_confidence=0.5):
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    candidates = prediction[prediction[:,:,4] > min_confidence].unsqueeze(0)
    batch_size, n_bboxes, pred_vector = candidates.size()
    for image in range(batch_size):
        top_class_per_bbox = torch.argmax(candidates[image, :, 5:], axis=1)
        unique_classes = torch.unique(top_class_per_bbox)
        candidates = torch.hstack([candidates[image, :, :5], top_class_per_bbox.unsqueeze(1)])

        all_ = []
        for idx, c in enumerate(unique_classes):
            top_bboxes_per_class = candidates[candidates[:,5] == c]
            keep = nms(top_bboxes_per_class, 0.5)
            tmp = top_bboxes_per_class[keep]
            for i in keep:
                tmp = top_bboxes_per_class[i]
                all_.append(torch.tensor([0, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], 1, tmp[5]]))
                
    return torch.stack(all_)
    

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (resize_with_ratio(img))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

