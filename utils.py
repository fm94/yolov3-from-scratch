from __future__ import division
import torch 
import numpy as np
import cv2 
import time

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

def get_all_bboxes(prediction, inp_dim, anchors, num_classes, threshold, use_gpu=False):
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
    #prediction = prediction[prediction[:,:,4] > threshold].unsqueeze(0)
    #keep = prediction.size()[1]
    
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
    # return only a few
    return prediction[prediction[:,:,4] > threshold].unsqueeze(0)

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

def get_final_bboxes(prediction, nms_threshold):
    # convert coordinates // this is apparently slow
    x = prediction[...,0].clone()
    y = prediction[...,1].clone()
    w = prediction[...,2].clone()
    h = prediction[...,3].clone()
    prediction[...,0] = x - w/2
    prediction[...,1] = y - h/2
    prediction[...,2] = x + w/2
    prediction[...,3] = y + h/2

    batch_size, n_bboxes, pred_vector = prediction.size()
    for image in range(batch_size):
        top_class_per_bbox = torch.argmax(prediction[image, :, 5:], axis=1)
        unique_classes = torch.unique(top_class_per_bbox)
        prediction = torch.hstack([prediction[image, :, :5], top_class_per_bbox.unsqueeze(1)])
        all_ = []
        for idx, c in enumerate(unique_classes):
            top_bboxes_per_class = prediction[prediction[:,5] == c]
            keep = nms(top_bboxes_per_class, nms_threshold)
            for i in keep:
                all_.append(top_bboxes_per_class[i])
    return torch.stack(all_) if all_ else None


def transform_image(img, input_dim):
    # reszing with ratio
    # maybe optimize this
    old_size = img.shape[:2]
    ratio = float(input_dim)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = input_dim - new_size[1]
    delta_h = input_dim - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    # normalization and permutation of axis
    return (torch.tensor(img/255, dtype=torch.float).permute((2,0,1)).unsqueeze(0), top, left, ratio)

def load_classes(path):
    with open(path) as file:
        labels = [line.rstrip() for line in file]
    return labels


def put_boxes_on_image(prediction, final_img, nms_threshold, left, top, ratio, box_color, text_color, classes):
    prediction = get_final_bboxes(prediction, nms_threshold)
    if not torch.is_tensor(prediction): return final_img
    
    # rescale bboxes
    prediction[...,[0, 2]] =  (prediction[...,[0, 2]] - left) / ratio
    prediction[...,[1, 3]] =  (prediction[...,[1, 3]] - top) / ratio
    # write boxes
    n_boxes, _ = prediction.size()
    for box in range(n_boxes):
        c1 = tuple([int(value) for value in prediction[box, :2]])
        c2 = tuple([int(value) for value in prediction[box, 2:4]])
        cv2.rectangle(final_img, c1, c2, box_color, 1)
        label = f"{classes[int(prediction[box, -1])]}: {prediction[box, -2]:.3f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(final_img, (c1[0], c1[1] - 20), (c1[0] + w, c1[1]), box_color, -1)
        cv2.putText(final_img, label, (c1[0], c1[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)
    return final_img