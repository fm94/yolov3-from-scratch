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

parser.add_argument("--images", default="imgs", type=str, 
                    help="Image / Directory containing images to perform detection upon",)
parser.add_argument("--output_dir", default="detections", type = str,
                    help ="Directory where to store annotated pictures")
parser.add_argument("--nms_threshold", default = 0.5, type = float,
                    help="max IOU to remove duplicates in NMS")
parser.add_argument("--config", default = "official_configs/yolov3.cfg", type = str,
                    help = "Network configuration file")
parser.add_argument("--classes", default = "data/class_names/coco.names", type = str,
                    help = "Network configuration file")
parser.add_argument("--weights", default = "official_weights/yolov3.weights", type = str,
                    help="Pre-trained network weight file")

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

# load images (see if file or dir)
if os.path.isfile(opt.images):
    all_image_paths = [opt.images]
elif os.path.isdir(opt.images):
    all_image_paths = [os.path.join(opt.images, fn) for fn in next(os.walk(opt.images))[2]]
else:
    print(">> images path is invalid...Exiting...")
    exit(1)
all_images = [cv2.imread(path) for path in all_image_paths]

# load classes
classes = utils.load_classes(opt.classes)

# do transformations including scaling and normalization
#all_batches = torch.stack([utils.transform_image(img, input_dim) for img in all_images])
all_batches = [utils.transform_image(img, input_dim) for img in all_images]

# put them into batches

# loop over batches
#for batch in torch.split(all_batches, int(len(all_images)/opt.batch_size)):

# loop over images
for idx, (image, top, left, ratio) in enumerate(all_batches):
    # get predictions without gradients then itss fast
    start = time.time()
    with torch.no_grad():
        prediction = model(image)
    output_time = time.time()
    # filter out bboxes
    final_img = utils.put_boxes_on_image(prediction, all_images[idx], opt.nms_threshold, left, top, ratio, box_color, text_color, classes)
    # save image
    file_name = os.path.join(opt.output_dir, "annotated_" + os.path.basename(all_image_paths[idx]))
    cv2.imwrite(file_name, final_img)
    finish_time = time.time()
    print(f">> Forward pass time:          {output_time - start:.3f}s")
    print(f">> Entire inference + io time: {finish_time - start:.3f}s")