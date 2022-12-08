import argparse
import os
import torch 
import cv2
import time
import numpy as np

import utils
import network

# TODOs
# implement batch-based inference
# currently loading a folder is not working

parser = argparse.ArgumentParser(description='Detect objects in a Picture using YoloV3')

parser.add_argument("--video", default="imgs", type=str, 
                    help="Image / Directory containing images to perform detection upon",)
parser.add_argument("--output_file", default="detections/output_video.avi", type = str,
                    help ="Directory where to store annotated pictures")
parser.add_argument("--nms_threshold", default = 0.5, type = float,
                    help="max IOU to remove duplicates in NMS")
parser.add_argument("--config", default = "official_configs/yolov3.cfg", type = str,
                    help = "Network configuration file")
parser.add_argument("--classes", default = "data/class_names/coco.names", type = str,
                    help = "Network configuration file")
parser.add_argument("--weights", default = "official_weights/yolov3.weights", type = str,
                    help="Pre-trained network weight file")
parser.add_argument("--save", action='store_true', help="Whether to store the annotated video into disk")

opt = parser.parse_args()

model = network.YoloV3(opt.config)
print(">> Loading Weights...")
model.load_weights(opt.weights)
print(">> Loaded")
config = model.network_info
assert config['width'] == config['height'], "Config file contains non-squared images"
input_dim = int(config['width'])

box_color = (255, 165, 0)
text_color = (255,255,255)

# load video
capture = cv2.VideoCapture(opt.video) 
if not capture.isOpened():
    print(">> Cannot read video file... Exiting...")
    exit(1)

if opt.save:
    width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(opt.output_file, fourcc, 25, (width, height))
    
# load classes
classes = utils.load_classes(opt.classes)

frames = 0
start = time.time()
while capture.isOpened():
    ret, frame = capture.read()
    if ret:  
        (modified_frame, top, left, ratio) = utils.transform_image(frame, input_dim)
        with torch.no_grad():
            prediction = model(modified_frame)
        final_img = utils.put_boxes_on_image(prediction, frame, opt.nms_threshold, left, top, ratio, box_color, text_color, classes)
        
        if opt.save: video.write(final_img)
        cv2.imshow("frame", final_img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break
cv2.destroyAllWindows()
video.release()