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

The interface details of 'MultiObjcetTracker' is explained in this repository: https://github.com/asujaykk/MultiObjectTracker.git

## 3. How to run object tracker demo?
This object tracker is designed to work with 'yolov7 object detector', but it can be used with any object detector by formating the output of object detector to make it compatable with 'tracker.track()' method.
The expected parameter format of 'tracker.track()' method explained here(https://github.com/asujaykk/MultiObjectTracker.git). 
So if you are using any other object detection model then please convert the detection output to the supported format before passing to 'tracker.track()' method.

1. As a first step clone this repository (its a stripped copy of  'https://github.com/WongKinYiu/yolov7.git' repository only for inferenec) to your working directory with below command.
```
  git clone https://github.com/asujaykk/Yolov7-Multi-Object-tracking.git
```

2. Then create an anaconda virtual environment and install required packages with "requirements.txt". Basically most of the packages are required for yolov7 inference. Or you can use the "yoloMOT_condaenv_bckp.yaml" file to create the anaconda virtual environment easly with below command.  

```
conda env create -f yoloMOT_condaenv_bckp.yaml
```
The created virtual environment name will be "python37gpu".

4. Then activate the anaconda environment with following command.
```
conda activate python37gpu
```
5. Then enter the 'Yolov7-Multi-Object-tracking' directory with following command,
```
 cd Yolov7-Multi-Object-tracking
```

The tracker can be used in two modes they are:
1. ***Normal tracking mode:***   
   All available objects types/classes will be tracked.
2. ***Selective tracking mode:***  
   In this mode user can configure a list of classes to be tracked, then the tracker skip all other objects classes from tracking. This will help if you are concerened about only a set of object types.  
   ex: Only monitor persons in a scene, only monitor cars from traffic , only monitor 'cars and busses and motorcycles' from the scene.

### 3.1. Normal tracking mode. 
To use the tracker in normal mode, please set 'sel_classes' variable  to 'None' in 'object_tracking_v2.py' as shown below.
![Screenshot_20230114-182546](https://user-images.githubusercontent.com/78997596/212472599-7e27714d-1363-4d5d-9ee4-021891a37da0.jpg)


Then run the following command to start yolov7 MO-tracker,
```
python3 object_tracking_v2.py --source <path_to_video_file or 0 for webcam> --view-img
```
In this mode the tracker try to track all available objects in the scene. A small GIF provided here to show the result.The interface of 'object_tracking_v2.py' is adapted from "detect.py" of  'https://github.com/WongKinYiu/yolov7.git' repo.

![20230114_180531](https://user-images.githubusercontent.com/78997596/212471881-ef36965b-9b33-4224-88c2-3b04d7b43b0f.gif)



### 3.2. Selective tracking mode. 
To use the tracker in Selective mode, please set 'sel_classes' variable  to ['truck'] in 'object_tracking_v2.py' as shown below.

![Screenshot_20230114-182439](https://user-images.githubusercontent.com/78997596/212472643-e1877830-59a7-4740-8980-1e65b0e45a80.jpg)

Then run the following command to start yolov7 MO-tracker,
```
python3 object_tracking_v2.py --source <path_to_video_file or 0 for webcam> --view-img
```
In this mode the tracker track only 'truck' in the scene. A small GIF provided here to show the result.The interface of 'object_tracking_v2.py' is adapted from "detect.py" of  'https://github.com/WongKinYiu/yolov7.git' repo.

![20230114_180801](https://user-images.githubusercontent.com/78997596/212471897-14de6a24-1c21-4a09-a978-7eacb830c811.gif)



## Applications.
1. Traffic monitoring.
2. Vehicle speed detection from traffic camera.
3. Accident detection.
4. Environment monitoring for self driving cars.
5. Object tracking drones.



