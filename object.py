#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 21:49:55 2022

@author: akhil_kk
"""
import math
import numpy as np

class object:
    """
    The object class for handling each object
       
    """
    prev_f=None
    curr_f=None
    count=None
    #the threshold for match_score
    
    match_thresholds=np.array([0.5,1.00,0.50])  # dummy value for matching threshold 
   
    def __init__(self,img,bbox,conf,cls):
         """
        Object constructor
        Parameters
        ----------
        img : numpy array
            The current image which is being processed
        bbox : tuple (x,y,x,y)
            boundingbox of the object in the image
        conf : float
            class confidence of this object
        cls : int
            represent the class id of the object

        Returns
        -------
        None.

        """
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
        """
        This method will check for object match
        First the class will be compared
        1. if the classes are matching then proceed for iou and dist calculation (because iou and dist claculation introduce a bit overhead)
        2. If the class is not matching then provide a non match output
        
        numpy array of match score will be returned. match score format =[class_match,distance,IOU]
        
        """
        
        if self.cls==obj.cls:       #if the classes are matching then proceed for iou and dist calculation (because iou and dist claculation introduce a bit overhead)
           cls_match=0.00
           dist=object.get_closeness(self.centre, obj.centre)
           iou= object.get_iou(self.bbox,obj.bbox)

        else:
           cls_match=1.00   # cls_match=1 indicate not matching
           dist=500.00      # dist= distance between the two object (a large value to indicate not matching)
           iou=1.00         # iou = inverted intersection of union (1-iou)

        
        #result.append(pred_match)
        return np.array([cls_match,dist,iou],dtype=np.float32)  # numpy array of match score will be returned
        
        
    
    def update_object(self,obj):
        """
        The old objects will be updated from the new detections

        Parameters
        ----------
        obj : object
            The object from which the parameters to be updated.

        Returns
        -------
        None.

        """
        #save old properties to hist
        #self.hist=np.append(self.hist,obj.hist,axis=0)
        
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
    def initialize_matcher_threshold(cls,match_threshold):
        cls.match_thresholds=match_threshold   # to update the threshold limit of object class

    @staticmethod
    def get_iou(boxA,boxB):
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
        
    	# determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return 1.00-iou   # maximum value is 1 : we are inverting iou value so that for best match the value will be small
    
    
    @staticmethod
    def get_closeness(x0y0,x1y1):
         """
         This method will check the closeness of  centres of two objects
         x0y0: centre (x,y) of object0
         x1y1:centre (x,y) of object1
    
              
         returns distance between the two points (in pixels)
         """  
         dist=math.sqrt( ((x1y1[0] - x0y0[0])**2) + ((x1y1[1] - x0y0[1])**2) )
         #dist=math.dist(x0y0,x1y1) #only in python 3.8 onwards
     
         return dist    
      
    
    @staticmethod
    def is_matching(match,threshold):
        """
        check whether the match score is within the threshold limit
        
        1. first check for class match if they are not matching then no need to compare the remaining  (to save time)
        
        Return True if the match score is withing the threshold limit
        return False if the match score not withing the threshold limit
        """
        
        if match[0]<threshold[0]: # class matching
        
            if np.all(np.less(match,threshold)): # if class matches then check for all matchings
                return True
            else:
                return False
        else:
             return False
         
    @staticmethod
    def get_best_match(match1,match2): 
        """
        

        Parameters
        ----------
        match1 : numpy array of object match score
            match score of object1
        match2 : numpy array of object match score
            match score of object2

        Returns
        -------
        int
            return an integer representing the best match
            0 or 1 indicate match1 is best (minimum value for all matches)
            2 indicate match 2 is best 

        """
         
        scr1=np.sum(match1)
        scr2=np.sum(match2)
        if scr1<scr2:
            return 1
        elif scr2<scr1:
            return 2
        else:
            return 0
               
            
        

    