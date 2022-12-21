#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:13:23 2022

@author: akhil_kk
"""

"""
This is the tracker class which is used to tracke the object detected by the yolo detector


The constructor accept two parameter:
    1. The class name list as input
    2. maximum frame count to keep missed objects (default=7 frames)

The track method accept two parameter
    1. Input image
    2. The detection output of the detector (bbox,confidence,class)
       bbox=[x1,y1,x2,y2]
           
           x1,y1= lower left corner
           x2,y2= upper top corner
           
    IMPORTANT:        x2>x1
                      y1>y2
"""

import cv2  
import copy
import numpy as np
import math 

class tracker:
    
    def __init__(self,names,mfc=7):
        """
        The constructor accept two parameter:
            1. The class name list as input
            2. maximum frame count to keep missed objects (default=7 frames)
        """
        self.mfc=mfc                 
        self.names=names
        self.class_count=len(names)   # the label count
        self.fr_count=0               # Frame count to keep track of the tracking.will be incrimented on each track method call
        
        
        self.dummy_objects=[]        
        for _ in range(self.class_count):  # create a dummy list of dictionary for each class
            self.dummy_objects.append({})
            
        
        
        self.old_objects=copy.deepcopy(self.dummy_objects) # create old object dictionary list
        self.new_objects=copy.deepcopy(self.dummy_objects) # create new object dictionary list
        
        
        self.count=np.zeros(self.class_count,dtype=np.uint)  #create a numpy array with class size (with zero intialized) used to keep the new label for any new object for each class.
        self.temp_count=copy.deepcopy(self.count)            #create a numpy array with class size (with zero intialized) used to keep the new label for any new object for each class.
        
    def track(self,im0,det):   
        """
        this method will track the object 
        Parameters
        ----------
        im0 :  numpy array
            Input image 
        det : list
            The detection output of the detector (bbox,confidence,class)
               bbox=[x1,y1,x2,y2]
                   
                   x1,y1= lower left corner
                   x2,y2= upper top corner
                   
            IMPORTANT:        x2>x1
                              y1>y2

        Returns
        -------
        result : List (detction)
            The detetion with tracked labels
            [bbox,label(string),class]

        """
        #The new object dictionary should be cleared after 10 frames to remove missed objects 
        #the following if condition take care of that 
        
        if self.fr_count>=self.mfc:    
          self.new_objects=copy.deepcopy(self.dummy_objects)
          self.fr_count=0
        self.fr_count+=1
           
        # Write results
        result=[]
        for *xyxy, conf, cls in reversed(det):
            label = self.names[int(cls)]
            #print("AKHIL")
            #print(int(cls))
            #print(im0.shape)
            
            #x is bounding box of the detected object
            #c1 : lower left corner point of bbox
            #c2 : upper right corner of bbox
            #centre : centre point of bbox
            x=xyxy
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) 
            centre=(c1[0]+((c2[0]-c1[0])//2),c2[1]+((c1[1]-c2[1])//2))
            
            # dynamic pixel threshold (bbox diagonal distance) for near by object search 
            # The search area will be reduced if the object size reduce/object moving far away in the scene
            threshold=math.sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 ) 
            
            #im0=cv2.circle(im0, centre, 5, (255,50,50), 5)
            r_id=0  # r_id=0 means new object other wise it will be updated by get_label method
            if len(self.old_objects[int(cls)])==0:   # for first frame or if there is no entry was stored for a particular class earlier then store all objects with distict ids
                self.temp_count[int(cls)]+=1         # Store id for the current objects
                self.new_objects[int(cls)].update({self.temp_count[int(cls)]:centre})  # add new object with id and centre
                
            else:  # if there is an entry in old object  
                r_id=self.get_label(int(cls),centre,threshold)  # serach for nearest object based on the current object centre
                if r_id==0:                                     # if the returned id==0 then this object is a new one
                    self.temp_count[int(cls)]+=1                # incriment class countas for a new entry
                    r_id=self.temp_count[int(cls)]              # change r_id to new id
                    self.new_objects[int(cls)].update({self.temp_count[int(cls)]:centre})  #add new object
                else:                                           # if the object had a match from the old objects
                   #print(cls)
                   #print(count[int(cls)])
                   #print(old_objects[int(cls)].keys())
                   self.old_objects[int(cls)].pop(r_id)         # remove the identified object from old object to avoid multiple match with other near objects
                   
                   self.new_objects[int(cls)].update({r_id:centre}) # copy old object id to new object with new location 
                #print(count)

            label=self.names[int(cls)]+" "+str(r_id)   # create label string
            result.append((xyxy,label,int(cls)))       # append the result for each detection
        self.old_objects=copy.deepcopy(self.new_objects)  # replace old objects with new objects
        self.count=copy.deepcopy(self.temp_count)      # replace class count with new class counts from temp_count
        return result  # return detection with new id for plotting
    
    def is_close(self,x0y0,x1y1,threshold):
        """
        This method will check the closeness of  centres of two objects
        x0y0: centre (x,y) of object0
        x1y1:centre (x,y) of object1
        threshold: The maximum distance (in pixels) two points can have, if the centres are greater than this threshold then these objects are considered as different.
             
        returns boolean:
            True : if the object centres are close
            False : if the objects are not close enough
        """  
        dist=math.sqrt( (x1y1[0] - x0y0[0])**2 + (x1y1[1] - x0y0[1])**2 )
        #dist=math.dist(x0y0,x1y1) #only in python 3.8 onwards
 
        if dist<threshold:
            return True
        else:
            return False
    
    
    def get_label(self,cls,centre,threshold=25):
        """
        Identify the objects label from the old object record which is near to the centre point. 
        old_objects: The dictionary contain object label(integer) as key and object centre (tuple x,y) as value
        centre: The centre   
        
        returns:
            the key/ID of the object from old object dictionary which shows best match
            If there is no match found then the returned label will be '0'
        """

        th=threshold   # the maximum distance allowed in pixel
        th_l=1  # the minimum distance  allowed to distinguish two objects
        match_count=0  #the variable used to check whether the centre matching two objects or not
        label=0   # 
        no_stop=True  #The variable used to stop the while iteration
        while no_stop and th>th_l: 
            for key,value in self.old_objects[cls].items():   # the for loop used to check presence of close objects
                
                if self.is_close(value,centre,th_l):   # if close object found
                    #pass
                    label=int(key)  # get the label of close object
                    no_stop=False
                    break
            else:                   # if for loop completed without any entry in the dictionary, then it is time to repeat the search with increased range
               th_l+=1        
        
        #if label==-1:                      # if no label found in the dictionary
          #label=max(self.old_objects[cls].keys())+1  # Then add a new key , which is not present in the dict
          #print("new_label="+str(label))                 

        return int(label)              