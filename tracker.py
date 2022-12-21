#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:50:29 2022

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
            x=xyxy
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            centre=(c1[0]+((c2[0]-c1[0])//2),c2[1]+((c1[1]-c2[1])//2))
            #im0=cv2.circle(im0, centre, 5, (255,50,50), 5)
            r_id=0
            if len(self.old_objects[int(cls)])==0:
                self.temp_count[int(cls)]+=1
                self.new_objects[int(cls)].update({self.temp_count[int(cls)]:centre})
                
            else:
                r_id=self.get_label(int(cls),centre)
                if r_id==0:
                    self.temp_count[int(cls)]+=1
                    r_id=self.temp_count[int(cls)]
                    self.new_objects[int(cls)].update({self.temp_count[int(cls)]:centre})
                else:  
                   #print(cls)
                   #print(count[int(cls)])
                   #print(old_objects[int(cls)].keys())
                   self.old_objects[int(cls)].pop(r_id)  
                   
                   self.new_objects[int(cls)].update({r_id:centre})
                #print(count)

            label=self.names[int(cls)]+" "+str(r_id)
            result.append((xyxy,label,int(cls)))
        self.old_objects=copy.deepcopy(self.new_objects)
        self.count=copy.deepcopy(self.temp_count)
        return result
    
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
    
    
    def get_label(self,cls,centre):
        """
        Identify the objects label from the old object record which is near to the centre point. 
        old_objects: The dictionary contain object label(integer) as key and object centre (tuple x,y) as value
        centre: The centre   
        
        returns:
            the key/ID of the object from old object dictionary which shows best match
            If there is no match found then the returned label will be '0'
        """

        th=25   # the maximum distance allowed in pixel
        th_l=1  # the minimum distance  allowed to distinguish two objects
        match_count=0  #the variable used to check whether the centre matching two objects or not
        label=0   # 
        no_stop=True  #The variable used to stop the while iteration
        while no_stop and th>th_l: 
            for key,value in self.old_objects[cls].items():   # the for loop used to check presence of close objects
                
                if self.is_close(value,centre,th):   # if close object found
                    #pass
                    label=int(key)  # get the label of close object
                    match_count+=1  # incriment matchcount
                if match_count>1:   # if more than one object present near the centre point within the threshold limit.
                    #label=0       # reset label
                    th-=1           # reduce threshold value to reduce closeness range for repeated search in the dictiory.   
                    break           # break for next search with reduced closeness range
            else:                   # if for loop completed without multiple entry in the dictionary, then it is time to break the search
                no_stop=False       
        
        #if label==-1:                      # if no label found in the dictionary
          #label=max(self.old_objects[cls].keys())+1  # Then add a new key , which is not present in the dict
          #print("new_label="+str(label))                 

        return int(label)              