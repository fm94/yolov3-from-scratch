from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

import utils

N_IN_CHANNELS = 3 # RGB
LEAKY_RELU_SLOP = 0.1

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        
def get_conv_block(block, index, layer_config, prev_filters):
    "Build a convolutional block from a config"
    if 'batch_normalize' in layer_config:
        batch_normalize = int(layer_config["batch_normalize"])
        bias = False
    else:
        batch_normalize = None
        bias = True
    filters= int(layer_config["filters"])
    kernel_size = int(layer_config["size"])
    stride = int(layer_config["stride"])
    padding = int(layer_config["pad"])
    activation = layer_config["activation"]
    pad = (kernel_size - 1) // 2 if padding else 0 # TODO: check this
    block.add_module(f"conv_{index}", nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias))
    if batch_normalize:
        block.add_module(f"batch_norm_{index}", nn.BatchNorm2d(filters))
    # currently conv layers use only learky relu
    if activation == "leaky":
        block.add_module(f"leaky_{index}", nn.LeakyReLU(LEAKY_RELU_SLOP, inplace = True))
    return block

def get_upsampling_block(block, index, layer_config):
    stride = int(layer_config["stride"])
    block.add_module(f"upsample_{index}", nn.Upsample(scale_factor = stride, mode = "nearest"))
    return block

def get_route_block(block, index, layer_config, output_filters):
    "TODO check this"
    routes = layer_config["layers"].split(',')
    start = int(routes[0])
    end = int(routes[1]) if len(routes) > 1 else 0
    if start > 0: 
        start = start - index
    if end > 0:
        end = end - index
    block.add_module(f"route_{index}", EmptyLayer())
    if end < 0:
        filters = output_filters[index + start] + output_filters[index + end]
    else:
        filters= output_filters[index + start]
    return block, filters

def get_yolo_block(block, index, layer_config):
    masks = [int(x) for x in layer_config["mask"].split(",")]
    anchors = [int(x) for x in layer_config["anchors"].split(",")]
    # select only tuples of anchors that are in masks
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2) if i in masks]
    block.add_module("Detection_{index}", DetectionLayer(anchors))
    return block  
      
def create_net(layers_config):   
    all_blocks = nn.ModuleList()
    prev_filters = N_IN_CHANNELS
    output_filters = []
    for index, layer in enumerate(layers_config):
        block = nn.Sequential()
        
        if layer["type"] == "convolutional":
            block = get_conv_block(block, index, layer, prev_filters)

        elif layer["type"] == "upsample":
            block = get_upsampling_block(block, index, layer)
                
        elif layer["type"] == "route":
            block, output_filter = get_route_block(block, index, layer, output_filters)
    
        elif layer["type"] == "shortcut":
            block.add_module(f"shortcut_{index}", EmptyLayer())

        elif layer["type"] == "yolo":
            block = get_yolo_block(block, index, layer)
                              
        all_blocks.append(block)
        prev_filters = output_filter
        output_filters.append(output_filter)
    return all_blocks

if __name__ == "__main__":
    layers_config, network_info = utils.parse_config('yolov3.cfg')