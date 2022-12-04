from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

import utils

# TODOs
#  avoid rewriting config twice. do it once when you build the network

N_IN_CHANNELS = 3 # RGB
LEAKY_RELU_SLOP = 0.1

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        
def get_conv_block(index, layer_config, prev_filters):
    "Build a convolutional block from a config"
    block = nn.Sequential()
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
    config = {}
    config['type'] = layer_config['type']
    config['batch_normalize'] = batch_normalize
    config['bias'] = bias
    config['filters'] = filters
    config['kernel_size'] = kernel_size
    config['stride'] = stride
    config['padding'] = [padding, pad]
    config['activation'] = activation
    return block, filters, config

def get_upsampling_block(index, layer_config):
    block = nn.Sequential()
    stride = int(layer_config["stride"])
    block.add_module(f"upsample_{index}", nn.Upsample(scale_factor = stride, mode = "nearest"))
    config = {}
    config['type'] = layer_config['type']
    config['stride'] = stride
    return block, config

def get_route_block(index, layer_config, output_filters):
    block = nn.Sequential()
    routes = [int(x) for x in layer_config["layers"].split(',')]
    filters = output_filters[index + routes[0]]
    if len(routes) > 1:
        # in case there is a second route take the direct index without offset
        filters += output_filters[routes[1]]
    block.add_module(f"route_{index}", EmptyLayer())
    config = {}
    config['type'] = layer_config['type']
    config['routes'] = routes
    return block, filters, config

def get_yolo_block(index, layer_config):
    block = nn.Sequential()
    masks = [int(x) for x in layer_config["mask"].split(",")]
    anchors = [int(x) for x in layer_config["anchors"].split(",")]
    # select only tuples of anchors that are in masks
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2) if i in masks]
    block.add_module("Detection_{index}", DetectionLayer(anchors))
    config = {}
    config['type'] = layer_config['type']
    config['masks'] = masks
    config['anchors'] = anchors
    ...
    return block, config

def get_shortcut_block(index, layer_config):
    block = nn.Sequential()
    block.add_module(f"shortcut_{index}", EmptyLayer())
    config = {}
    config['type'] = layer_config['type']
    config['from'] = int(layer_config['from'])
    return block, config

def create_net(layers_config):   
    all_blocks = []
    prev_filters = N_IN_CHANNELS
    output_filters = []
    for index, layer in enumerate(layers_config):
        if layer["type"] == "convolutional":
            block, output_filter, config = get_conv_block(index, layer, prev_filters)
        elif layer["type"] == "upsample":
            block, config = get_upsampling_block(index, layer) 
        elif layer["type"] == "route":
            block, output_filter, config = get_route_block(index, layer, output_filters)
        elif layer["type"] == "shortcut":
            block, config = get_shortcut_block(index, layer)
        elif layer["type"] == "yolo":
            block, config = get_yolo_block(index, layer)               
        all_blocks.append([block, config])
        prev_filters = output_filter
        output_filters.append(output_filter)
    return all_blocks

class YoloV3(nn.Module):
    def __init__(self, config_file):
        super(YoloV3, self).__init__()
        self.network_info, self.layers_config = utils.parse_config(config_file)
        self.all_blocks = create_net(self.layers_config)
    
    def forward(self, x, CUDA=False):
        all_outputs = {}
        write = 0
        for index, (block, config) in enumerate(self.all_blocks):
            layer_type = config["type"]

            if layer_type == "convolutional" or layer_type == "upsample":
                x = block(x)
    
            elif layer_type == "route":
                routes = config["routes"]
                if len(routes) > 1:
                    first_route = all_outputs[index + routes[0]]
                    second_route = all_outputs[routes[1]]
                    x = torch.cat((first_route, second_route), 1)
                else:
                    x = all_outputs[index + routes[0]]
                    
            elif  layer_type == "shortcut":
                x = all_outputs[index-1] + all_outputs[index+config['from']]
    
            elif layer_type == 'yolo':        
                anchors = self.all_blocks[index][0].anchors
                inp_dim = int (self.network_info["height"])
                num_classes = int (layer["classes"])
                x = x.data
                x = utils.predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            all_outputs[index] = x
        
        return 
        
if __name__ == "__main__":
    layers_config, network_info = utils.parse_config('yolov3.cfg')
    print(create_net(layers_config))