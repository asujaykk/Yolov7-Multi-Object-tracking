#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:49:55 2022

@author: akhil_kk
"""
import math
import numpy as np

class object:
    
    prev_f=None
    curr_f=None
    count=None

   
    def __init__(self,img,bbox,conf,cls):
         
         self.bbox=bbox
         self.cls=cls
         c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
         self.c1=c1
         self.c2=c2
         centre=(c1[0]+((c2[0]-c1[0])//2),c1[1]+((c2[1]-c1[1])//2))
         self.centre=centre
         # dynamic pixel threshold (bbox diagonal distance) for near by object search 
         # The search area will be reduced if the object size reduce/object moving far away in the scene
         threshold=(math.sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 ))/2.00
         self.threshold=threshold

         #store the history of the object
         self.hist=np.array([[c1,c2,centre,threshold,cls]])     
         
    def get_match_score(self,obj):
        
        if self.cls==obj.cls:
           cls_match=0.00
        else:
           cls_match=1.00
        dist=object.get_closeness(self.centre, obj.centre)
        iou= object.get_iou(self.bbox,obj.bbox)

        
        #result.append(pred_match)
        return np.array([dist,cls_match],dtype=np.float32)
        
        
    
    def update_object(self,obj):
        #save old properties to hist
        self.hist=np.append(self.hist,[[obj.c1,obj.c2,obj.centre,obj.threshold,obj.cls]],axis=0)
        
        #copy properties from new obj to existing obj
        self.bbox=obj.bbox
        self.c1=obj.c1
        self.c2=obj.c2
        self.centre=obj.centre
        self.threshold=obj.threshold
    
      
    def get_hist_match(self):
        pass



    def get_opt_flow_match(self):
        pass


    def get_pred_loc(self):       
        if np.shape(self.hist)[0]>10:
            return None
        else:
            return None


    @classmethod
    def initialize_object_parameters(class_count):
        count=np.zeros(class_count,dtype=np.uint)  #create a numpy array with class size (with zero intialized) used to keep the new label for any new object for each class.
        pass


    @staticmethod
    def get_iou(bbox1,bbox2):
        """
        Parameters
        ----------
        bbox1 : tuple (x1,y1,x2,y2)
            bounding box of object1
            x1,y1= upper left corner
            x2,y2= Lower right corner
        bbox2 : tuple (x1,y1,x2,y2)
            bounding box of object2
            x1,y1= upper left corner
            x2,y2= Lower right corner
    
        Returns
        -------
        iou : float (value between 0-1)
            value indicate the intersection of union between two bounding boxes
    
        """
        obj1_area=abs(bbox1[2]-bbox1[0])*abs(bbox1[3]-bbox1[1])
        obj2_area=abs(bbox2[2]-bbox2[0])*abs(bbox2[3]-bbox2[1])
        
        inter_ul_xy=(max(bbox1[0],bbox2[0]),max(bbox1[1],bbox2[1]))
        inter_lr_xy=(min(bbox1[2],bbox2[2]),min(bbox1[3],bbox2[3]))
        
        inter_area=abs(inter_lr_xy[0]-inter_ul_xy[0])*abs(inter_lr_xy[1]-inter_ul_xy[1])
        
        iou=inter_area/(obj1_area+obj2_area-inter_area)
        #return 1.00-iou  # maximum value is 1 : we are inverting iou value so that for best match the value will be small
        return 1-iou
    
    @staticmethod
    def get_closeness(x0y0,x1y1):
         """
         This method will check the closeness of  centres of two objects
         x0y0: centre (x,y) of object0
         x1y1:centre (x,y) of object1
    
              
         returns distance between the two points (in pixels)
         """  
         dist=math.sqrt( (x1y1[0] - x0y0[0])**2 + (x1y1[1] - x0y0[1])**2 )
         #dist=math.dist(x0y0,x1y1) #only in python 3.8 onwards
     
         return dist    
    
    
    
    @staticmethod
    def is_matching(match,threshold):
        if np.all(np.less(match,threshold)):
            return True
        else:
            return False
            
    @staticmethod
    def get_best_match(match1,match2):
        scr1=np.sum(match1)
        scr2=np.sum(match2)
        if scr1<scr2:
            return 1
        elif scr2<scr1:
            return 2
        else:
            return 0
               
            
        

    