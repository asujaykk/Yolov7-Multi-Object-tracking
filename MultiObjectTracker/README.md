# Multi-Object-Tracker.
This is a multi object tracker package designed to use with any object detectors. We tested it with YOLOv7 object detector, but it will work with other detectors as well by formatting the output of the object detector to make it compatable with the tracker.

### Requirements:
1. opencv : 4.5.3.56 
2. Numpy  : 1.19.2 
3. Python  : 3.7.13 

## 1. Object tracking:
Object tracking is the process of identifying same object and keep track of their location with unique label as they move around in a video. Object tracker consist of two sections.
1. ***Object detector:***  
An object detector detects different objects , their locations (bounding box) and class type from a single video frame.
2. ***Tracker:***  
The tracker process the detections of the current frame and  identify the best matches for the objects from the previous frame. The matched objects will get the unique identification from the previous objects. The tracker also need to clear missed object and add new entries as video progress. 

## 2. Tracking process:
As mentioned before ,in object tracking the first step is to detect, locate (bounding box) and find class type of objects from a video frame. Then the detections from the detector will be  passed to the tracker as input and tracker will keep track of the detected objects with unique names/labels from frame to frame until the object un-detected/lost from the scene. The major steps involved in the tracking process is as follows. 
![Screenshot from 2023-01-08 16-36-57](https://user-images.githubusercontent.com/78997596/211193456-bf8bf9f8-c8ec-416d-aa93-5871e931103f.png)


## 3. How to use the tracker with object detection and recognition model:
This object tracker is designed to work with 'yolov7 object detector', but it can be used with any object detector by formating the output of object detector to make it compatable with 'tracker.track()' method.The expected parameter format of 'tracker.track()' method explained in following section. So if you are using any other object detection model then please convert the detection output to the supported format before passing to 'tracker.track()' method.

As a first step download 'MultiObjectTracker' package to your working directory with below command,
```
git clone git@github.com:asujaykk/MultiObjectTracker.git
```
Then import the tracker module (tracker_v3.py) to your python code as follows,

```
from MultiObjectTracker.tracker_v3 import tracker 
```

The tracker can be used in two modes they are:
1. ***Normal tracking mode:***   
   All available objects types/classes will be tracked.
2. ***Selective tracking mode:***  
   In this mode user can configure a list of classes to be tracked, then the tracker skip all other objects classes from tracking. This will help if you are concerened about only a set of object types.  
   ex: Only monitor persons in a scene, only monitor cars from traffic , only monitor 'cars and busses and motorcycles' from the scene.

### 3.1. Normal tracking mode. 
The following general code template shows how to use the tracker in normal mode with an objet detection and recognition code,

   ```
   #Import modules
   from MultiObjectTracker.tracker_v3 import tracker   #load tracker module 
   import object_detector                              #just a template only
 
   #Create tracker object with list of class names as constructor parameter. 
   mo_tracker=tracker(list_class_names,sel_classes=None,mfc=10,max_dist=None)  #sel_classes=None enable Normal tracking mode.
       
   #loop through  video frames
   for frame in video_frames:
       detections=object_detector(frame)                # process video frames one by one
       tracker_out=mo_tracker.track(frame,detections)   #Invoke **tracker.track(im0,det)** method for starting the tracking process.
       plotting_function(frame,tracker_out)             # tracker output will be used for plotiing/other applications
       visualizer(frmae)                                #visualizing function
   ```
   
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
   
### 3.3 The tracker constructor:
The tracker cnstructor accept four parameters and they are explained below.   
   1. names       : The list of class names supported by the object detector (list of strings, their index indicate the class label)        
   2. sel_classes    : If user want to track only a particular class objects then the list of classes to be tracked can be provided here.   
      List of class labels to be tracked (list of class names (string) to be tracked)   
      If it is 'None' then tracker track all available objects , default value is None.         
      example: ['car','truck','person'] Then tracker only track 'car', 'truck' and 'person' only.              
   3. mfc         : maximum frame count to keep missed objects (default=10 frames)
   4. max_dist    : maximum distance (in pixels : integer) of movement object centre allowed between consequtive frames to consider they are same object.
      if 'None', then dynamic threshold will be used (the threshold will be set based on the previous object size )
      
### 3.4 The tracker.track() method:
The 'tracker.track(im0,det)' method accept two parameters and they are as follows.  
   1. im0 : The frame/image being processed (it should be a numpy array)  
   2. det : List of detections (from the detector) of one frame. 
   A single detection should have (bbox,confidence,class) format.              
   * bbox=bounding box of the object[x1,y1,x2,y2]   x1,y1= upper left corner , x2,y2= Lower right corner  
   * confidence= confidence of the object detector (float value)  
   * cls= integer ,represent the index position of the class in the class name list.

 and return a list of the detections with label for each object.A single returned detection have (bbox,conf,class,label) format.  
   * bbox=bounding box of the object[x1,y1,x2,y2]  x1,y1= upper left corner , x2,y2= Lower right corner 
   * conf= confidence of the detection (conf of the detector).
   * class= integer representing the class.*
   * label= string representing an objecrt "classname_<id>"   example: truck_5, car_6.

## 4. Methods used for object matching between current frame and previous frame:
The objects are matched using the following concepts, their corresponding methods can be found in object class ("object.py").
1. Nearest object search.
2. Iou matching.
3. Histogram matching.

### 4.1 Customizing/adding matching methods:
If user want to add more image matching method to improve the matching process, then user can add their own matching methods under 'object' class.
The method should meet the following requirements,
1. It should be a static method defined under 'object' class.
2. The method should return a match score (float value) between '0' and '1', where value close to '0' indicate best match and value close to '1' indicate least match.
3. The matching method should be invoked from 'get_match_score' method of 'object' class. and the resulting match score should be appended in return value (numpy array) of this method.

Note: 
1. Do not introduce many methods since it slows down the tracking process. 
2. Please ensure that the processing time of your algorithm is very small otherwise it affect over all performance of the tracker.

## 5. Applications.
1. Traffic monitoring.
2. Vehicle speed detection from traffic camera.
3. Accident detection.
4. Environment monitoring for self driving cars.
5. Object tracking drones.


         
         
