# Yolov7 Multi Object tracking
This is the demo of the multi Object tracker with yolov7 detector. 



### Tracking procedure.
1. Object detection and recognition.
2. Create new objects with new detections from the detector.
3. Identify the best match for each old objects from the new objects.
4. Idnetyfy new objects from the new detections
5. identify missed objects
6. propogate matched objects and new objects for next stage and remove missed object for more than 7 frmaes.
7. return the labels for detections (label represent the objects) back for plotting or further application.

## Methods used for tracking.
1. Nearest object search
2. Iou matching


## Major tracker modules of the MultiObjectTracker
There are two tarckers available in this package and they are v2 and v3. These trackers have similar interface and both can be used with different detectors. The tracker v2 is a lite version and use nearest search and first match approach to track the object. And tracker v3 is the upgraded version and which use multiple object matching methods and improved cross matching algorithm for identifying best matches.

### tracker_v2 :
Tracker v2 is the basic version. It purely works based on the nearest object search. The tracker search the nearest object based on the movement of object centre point in consecutive frames.

### tracker_v3 :
Tracker v3 is a bit advanced version of tracker v2 and it work based on the following approaches.
1. Nearest object search.
2. Iou matcthing.
3. Best cross matching algorithm.  
It is computationally expensive but provide much accurate tracking. This tracker handle detections as objects and the user can add their own matching algorithms in "object" class to achive customized tracking.

## Aplications.
1. Traffic monitoring.
2. Vehicle spped detection from traffic camera.
3. Accident detection.
4. Environment monitoring for self driving cars.
5. Object tracking drones.
