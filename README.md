# Yolov7 Multi Object tracking
This is the demo of the multi Object tracker with yolov7 detector. 



Tracking procedure
1. Object detection and recognition.
2. Create new objects with new detections from the detector.
3. Identify the best match for each old objects from the new objects.
4. Idnetyfy new objects from the new detections
5. identify missed objects
6. propogate matched objects and new objects for next stage and remove missed object for more than 7 frmaes.
7. return the labels for detections (label represent the objects) back for plotting or further application.
