# Yolov7 Multi Object tracking
Object tracking is the process of identifying same object and keep track of their location with unique label as they move around in a video. A multiobject tracker do this tracking for more than one object in a scene. 
This is the demo of how to use the 'MultiObjectTracker' package with yolov7 object detector. The same approach can be adapted for using this 'MultiObjectTracker' with other object detection models.

## 1. Object tracking:
Multi Object tracker consist of two sections.
1. ***Object detector:***  
An object detector detects different objects , their locations (bounding box) and class type from a single video frame. Here we are using yoloV7 detector.
This detector is capable of detecting 80 types(classes) of objects. And also it provide better detection and performance than ever before.

2. ***Tracker:***  
The tracker process the detections of the current frame and  identify the best matches for the objects from the previous frame. The matched objects will get the unique identification from the previous objects. The tracker also need to clear missed object and add new entries as video progress. We are using 'MultiObjcetTracker' package for this purpose. 
This tracker can be integrated to an object detection code with very mnimal code change and very minimal interface parameters.

The user instructions and interface details of 'MultiObjcetTracker' is explained in this repository: https://github.com/asujaykk/MultiObjectTracker.git

## 3. How to use the tracker with object detection and recognition model:
This object tracker is designed to work with 'yolov7 object detector', but it can be used with any object detector by formating the output of object detector to make it compatable with 'tracker.track()' method.
The expected parameter format of 'tracker.track()' method explained here(https://github.com/asujaykk/MultiObjectTracker.git). 
So if you are using any other object detection model then please convert the detection output to the supported format before passing to 'tracker.track()' method.

As a first step download this repository (its a stripped copy of  WongKinYiu/yolov7 repository only for inferenec) to your working directory with below command.
```
  git clone https://github.com/asujaykk/Yolov7-Multi-Object-tracking.git
```
or clone the latest yolov7 repository
```
  git clone https://github.com/WongKinYiu/yolov7.git
```

Then enter the detector folder,
```
 cd Yolov7-Multi-Object-tracking
   or 
 cd yolov7
```

Then download 'MultiObjectTracker' package to your working directory with below command,
```
git clone git@github.com:asujaykk/MultiObjectTracker.git
```

The tracker can be used in two modes they are:
1. ***Normal tracking mode:***   
   All available objects types/classes will be tracked.
2. ***Selective tracking mode:***  
   In this mode user can configure a list of classes to be tracked, then the tracker skip all other objects classes from tracking. This will help if you are concerened about only a set of object types.  
   ex: Only monitor persons in a scene, only monitor cars from traffic , only monitor 'cars and busses and motorcycles' from the scene.

### 3.1. Normal tracking mode. 
To use the tracker in 

   
### 3.2. Selective tracking mode. 
The following general code template shows how to use the tracker in selective tracking mode with an objet detection and recognition code,

   ```
   #Import modules
   from MultiObjectTracker.tracker_v3 import tracker   #load tracker module 
   import object_detector                              #just a template only
   selective_objects=['car','truck','person']                           # List of class labels. Only object belongs to this class types [2,3,4] get tracked. rest of the objects ignored.
   #Create tracker object with list of class names as constructor parameter. 
   mo_tracker=tracker(list_class_names,sel_classes=selective_objects,mfc=10,max_dist=None)
       
   #Loop through  video frames
   for frame in video_frames:
       detections=object_detector(frame)                # process video frames one by one
       tracker_out=mo_tracker.track(frame,detections)   #Invoke **tracker.track(im0,det)** method for starting the tracking process.
       plotting_function(frame,tracker_out)             # tracker output will be used for plotiing/other applications
       visualizer(frmae)                                #visualizing function
   ``` 

## Aplications.
1. Traffic monitoring.
2. Vehicle spped detection from traffic camera.
3. Accident detection.
4. Environment monitoring for self driving cars.
5. Object tracking drones.
