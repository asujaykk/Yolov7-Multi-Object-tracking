#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:00:57 2022

@author: akhil_kk
"""



import copy       

def is_close(x0y0,x1y1,threshold):
    
    if abs(x0y0[0]-x1y1[0])<threshold and abs(x0y0[1]-x1y1[1])<threshold:
        return True
    else:
        return False
    
def get_label(old_objects,centre):
    th=25
    th_l=7
    match_count=0
    label=-1
    no_stop=True
    while no_stop and th>th_l: 
        for key,value in old_objects.items():
            
            if is_close(value,centre,th):
                #pass
                label=int(key)
                match_count+=1
            if match_count>1:
                label=-1
                th-=1
                break
        else:
            no_stop=False
    
    if label==-1:  
      label=max(old_objects.keys())+1
      #print("new_label="+str(label))                 

    return label


def update_old_objects(old_objects,new_objects):
    pass


def detect(q,save_img=False):
    
    
    import time
    import argparse
    import time
    from pathlib import Path
    import numpy as np

    import cv2
    import torch
    import torch.backends.cudnn as cudnn
    from numpy import random


    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
    
    
    
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
    vid_path, vid_writer = None, None
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
    
    car=[]
    truck=[]
    bus=[]
    
    dummy_objects=[]
    for _ in range(len(names)):
        dummy_objects.append({})
        
    old_objects=copy.deepcopy(dummy_objects)
    new_objects=copy.deepcopy(dummy_objects)
    
    count=np.zeros(len(names),dtype=np.uint)
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
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        frames=[]
        
        if fr_count>=10:  
          new_objects=copy.deepcopy(dummy_objects)
        fr_count+=1
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            ident=np.zeros((80))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    ident[c.cpu().detach().numpy().astype('uint8')]=n.cpu().detach().numpy().astype('uint8')
                    #print(c.cpu().detach().numpy().astype('uint8'))
                   
                # Write results
                
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img :  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        print("AKHIL")
                        #print(int(cls))
                        #print(im0.shape)
                        x=xyxy
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        centre=(c1[0]+((c2[0]-c1[0])//2),c2[1]+((c1[1]-c2[1])//2))
                        im0=cv2.circle(im0, centre, 5, (255,50,50), 5)
                        
                        if len(old_objects[int(cls)])==0:
                            new_objects[int(cls)].update({count[int(cls)]:centre})
                            count[int(cls)]+=1
                        else:
                            count[int(cls)]=get_label(old_objects[int(cls)],centre)
                            new_objects[int(cls)].update({count[int(cls)]:centre})
                        #print(count)
                        label=names[int(cls)]+" "+str(count[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                #old_objects.update(new_objects)
                #old_objects=new_objects.copy()
                
                #print(old_objects)
                #print(new_objects) 
            #frames.append(ident)
            
            # Print time (inference + NMS)
            #print(det[:,-1])
            #print(ident)
            #car.append(ident[2])
            #truck.append(ident[7])
            #bus.append(ident[5])
            q.put(ident)
            #time.sleep(0.1)
            print(s);
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
        old_objects=copy.deepcopy(new_objects)
        print("#######")



    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


import multiprocessing


def plot_f(q):
    import time
    import matplotlib.pyplot as plt
    
    plt.ion()    
    fig,ax=plt.subplots(3,2,figsize=(6,6))
    car=[]
    truck=[]
    bus=[]
    while True:
        while not q.empty():
           ident=q.get()
           car.append(ident[2])
           truck.append(ident[7])
           bus.append(ident[5])
        else:
             print("q empty")
       
        item_c=len(car)
        if item_c>200:
            # car.pop(0)
            # bus.pop(0)
            # truck.pop(0)
            ov=item_c-200
            car=car[ov:]
            bus=bus[ov:]
            truck=truck[ov:]
        
        
        ax[0,0].plot(car,'r-')
        ax[0,0].set_ylabel("car")
        ax[0,0].set_xlabel("time")
        ax[0,1].boxplot(car)
        
        ax[1,0].plot(truck,'g-')
        ax[1,0].set_ylabel("truck")
        ax[1,1].boxplot(car)
        
        ax[2,0].plot(bus,'b-')
        ax[2,0].set_ylabel("bus")
        ax[2,1].boxplot(car)
        
        
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1) 
        ax[0,0].clear()
        ax[1,0].clear()
        ax[2,0].clear()
        ax[0,1].clear()
        ax[1,1].clear()
        ax[2,1].clear()



def task1(opt,q):
    import time

    import time
    from pathlib import Path
    import numpy as np

    import cv2
    import torch
    import torch.backends.cudnn as cudnn
    from numpy import random


    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(q)
                strip_optimizer(opt.weights)
        else:
            
            detect(q) 





if __name__ == '__main__':
    import argparse
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
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    q=multiprocessing.Queue()
    p1=multiprocessing.Process(target=task1,args=(opt,q,))
    p2=multiprocessing.Process(target=plot_f,args=(q,))
    
    p1.start()
    p2.start()
    print("process started")
    p1.join()
    p2.join()
  




    
