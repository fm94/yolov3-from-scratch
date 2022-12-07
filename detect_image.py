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

parser = argparse.ArgumentParser(description='Detect objects in a Picture using YoloV3')

parser.add_argument("--images", default="imgs", type=str, 
                    help="Image / Directory containing images to perform detection upon",)
parser.add_argument("--output_dir", default="detections", type = str,
                    help ="Directory where to store annotated pictures")
parser.add_argument("--confidence_threshold", default = 0.5, type = float,
                    help="Minimum objectivness to consider an object",)
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
    all_image_paths = [f for f in os.listdir(opt.images)]
else:
    print(">> images path is invalid...Exiting...")
    exit(1)
all_images = [cv2.imread(path) for path in all_image_paths]

# load classes
classes = utils.load_classes(opt.classes)

# do transformations including scaling and normalization
all_batches = torch.stack([utils.transform_image(img, input_dim) for img in all_images])

# put them into batches

# loop over batches
#for batch in torch.split(all_batches, int(len(all_images)/opt.batch_size)):

# loop over images
for idx, image in enumerate(all_batches):
    # get predictions without gradients then itss fast
    start_det_loop = time.time()
    with torch.no_grad():
        prediction = model(image)
    # filter out bboxes
    prediction = utils.get_final_bboxes_(prediction, opt.confidence_threshold, opt.nms_threshold)
    output_recast = time.time()
    # rescale bboxes
    
    # write boxes
    n_boxes, _ = prediction.size()
    final_img = all_images[idx]
    for box in range(n_boxes):
        c1 = tuple([int(value) for value in prediction[box, :2]])
        c2 = tuple([int(value) for value in prediction[box, 2:4]])
        cv2.rectangle(final_img, c1, c2, box_color, 1)
        label = f"{classes[int(prediction[box, -1])]}: {prediction[box, -2]:.3f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(final_img, (c1[0], c1[1] - 20), (c1[0] + w, c1[1]), box_color, -1)
        cv2.putText(final_img, label, (c1[0], c1[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)
    # save image
    file_name = os.path.join(opt.output_dir, "annotated_" + os.path.basename(all_image_paths[idx]))
    cv2.imwrite(file_name, final_img)
    
    print(output_recast - start_det_loop)