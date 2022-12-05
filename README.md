# yolov3-from-scratch
An attempt to implement YoloV3 with pytorch<br>
Paper: https://arxiv.org/pdf/1804.02767.pdf<br>
Architecture: https://miro.medium.com/max/720/1*d4Eg17IVJ0L41e7CTWLLSg.webp (source: https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)<br>

# Comments
The idea is to build a CNN that takes images split into a 13x13 boxes and outputs a feature map of the same size with with depth B * (5 + C) -> B is is the max number of bounding boxes per section, C is the number of classes and 5 are x,w,h,b, and confidence (objectivness). Thus, per output box we have o = [tx,ty,tw,th,p,c0,...cn]
- we use 416x416 input images with stride 32 thus, the output is 13x13 --total--> 13x13x(Bx(5+C)).
- in yolov3, we use 3 anchors, thus B=3.
- actually the outputs of the network are further transformed to obtain the real coordinates and scales.
    * bx = sig(tx) + offset_x
    * by = sig(ty) + offset_y
    * bw = pw exp(th) # pw is anchor length in w i.e., 13 - check this
    * bh = ph exp(th) # ph is anchor length in h i.e., 13 - check this
- in yolov3, we use sigmoid for the classes to allow for multilabels (e.g., person, woman). Before, softmax was used which doesn;t really allow for double (multi-label) predictions.
- multiscaling: after striding at 32, upsimple again and apply strides 16 and 8, thus getting different grids: 13x13, 26x26 and 52x52 -> this allows detecting smaller objects.
- total boundin boxes for a 416x416 image: (52x52 + 26x26 + 13x13) x 3 = 10647
    * to reduce this number we could: apply a threshold on objectivness, apply NMS (remove boxes predicting same class - keep highest).
- loading the weights of a pre-trained network is difficult. They are stored without any structure!