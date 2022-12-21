#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:00:57 2022

@author: akhil_kk
"""


"""
This script will track all detected object in the scene based on the location change of the objects in the scene

procedure:

  OBJECT detection:
     1. The objects are detected using yolov7 object detection model
     2. The bounding box and class details will be saved for tracking the objects
  Tracking:
     1. The tracker keeps tracks of the previous object location
     2. In next frame the new object ID will be derived from the previous object location (based on the nearest object location ) 


"""


import math


import copy       
import time

from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import argparse

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def is_close(x0y0,x1y1,threshold):
    """
    This method will check the closeness of  centres of two objects
    x0y0: centre (x,y) of object0
    x1y1:centre (x,y) of object1
    threshold: The maximum distance (in pixels) two points can have, if the centres are greater than this threshold then these objects are considered as different.
               
    """  
    dist=math.sqrt( (x1y1[0] - x0y0[0])**2 + (x1y1[1] - x0y0[1])**2 )
    #dist=math.dist(x0y0,x1y1)
    #if abs(x0y0[0]-x1y1[0])<threshold and abs(x0y0[1]-x1y1[1])<threshold:
    if dist<threshold:
        return True
    else:
        return False
    
def get_label(old_objects,centre):
    """
    Identify the objects label from the old object record which is near to the centre point. 
    old_objects: The dictionary contain object label(integer) as key and object centre (tuple x,y) as value
    centre: The centre   
    """
    #print("get label")
    #print(old_objects.keys())
    th=25   # the maximum distance allowed in pixel
    th_l=1  # the minimum distance  allowed to distinguish two objects
    match_count=0  #the variable used to check whether the centre matching two objects or not
    label=0   # label =0 not moified means not match found in from the object list, hence the current object is a new entry  
    no_stop=True  #The variable used to stop the while iteration
    while no_stop and th>th_l: 
        for key,value in old_objects.items():   # the for loop used to check presence of close objects
            
            if is_close(value,centre,th):   # if close object found
                #pass
                label=int(key)  # get the label of close object
                match_count+=1  # incriment matchcount
            if match_count>1:   # if more than one object present near the centre point.
                #label=0       # reset label
                th-=1           # reduce threshold value to reduce closeness range for repeated search in the dictionary.   
                break           # break for reduced closeness search
        else:                   # if for loop completed without multiple entry in the dictionary, then it is time to break the search
            no_stop=False       
    
    #if label==-1:                      # if no label found in the dictionary
      #label=max(old_objects.keys())+1  # Then add a new key , which is not present in the dict
      #print("new_label="+str(label))                 

    return int(label)              




def detect(save_img=False):
    
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    
    dummy_objects=[]
    for _ in range(len(names)):
        dummy_objects.append({})
        
    old_objects=copy.deepcopy(dummy_objects)
    new_objects=copy.deepcopy(dummy_objects)
    
    count=np.zeros(len(names),dtype=np.uint)   # this will incriment the count for each class to create new able for new objects
    temp_count=copy.deepcopy(count)
    fr_count=0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        pred = model(img, augment=opt.augment)[0]


        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        if fr_count>10:  
          new_objects=copy.deepcopy(dummy_objects)
          fr_count=0
        fr_count+=1
        
        # Process detections
        #print("AKHIL pred count:"+str(len(pred)))
        for i, det in enumerate(pred):  # detections per image
            #print("AKHIL det count:"+str(len(det)))
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                   
                # Write results
                
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    #print("AKHIL")
                    #print(int(cls))
                    #print(count[int(cls)])
                    #print(im0.shape)
                    #print(old_objects[int(cls)].keys())
                    x=xyxy
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    centre=(c1[0]+((c2[0]-c1[0])//2),c2[1]+((c1[1]-c2[1])//2))
                    im0=cv2.circle(im0, centre, 5, (255,50,50), 5)
                    r_id=0
                    if len(old_objects[int(cls)])==0:
                        temp_count[int(cls)]+=1
                        new_objects[int(cls)].update({temp_count[int(cls)]:centre})
                        
                    else:
                        r_id=get_label(old_objects[int(cls)],centre)
                        if r_id==0:
                            temp_count[int(cls)]+=1
                            r_id=temp_count[int(cls)]
                            new_objects[int(cls)].update({temp_count[int(cls)]:centre})
                        else:  
                           #print(cls)
                           #print(count[int(cls)])
                           #print(old_objects[int(cls)].keys())
                           old_objects[int(cls)].pop(r_id)  
                           
                           new_objects[int(cls)].update({r_id:centre})
                        #print(count)

                    label=names[int(cls)]+" "+str(r_id)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
        old_objects=copy.deepcopy(new_objects)
        count=copy.deepcopy(temp_count)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

   
    with torch.no_grad():
       if opt.update:  # update all models (to fix SourceChangeWarning)
           for opt.weights in ['yolov7.pt']:
               detect()
               strip_optimizer(opt.weights)
       else:
           
           detect() 
    
  




    
