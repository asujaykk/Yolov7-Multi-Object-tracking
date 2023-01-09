#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:13:23 2022

@author: akhil_kk
"""

"""
This is the tracker class which is used to track the object detected by the yolo detector


The constructor accept two parameter:
    1. The class name list as input
    2. maximum frame count to keep missed objects (default=7 frames)
    3. max_dist: maximum distance (in pixels) of movement object centre allowed between consequtive frames to consider they are same object.
       if None: dynamic threshold will be used (the threshold will be set based on the previous object size ) 

The track method accept two parameter
    1. Input image
    2. The detection output of the detector (bbox,confidence,class)
       bbox=[x1,y1,x2,y2]
           
           x1,y1= upper left corner
           x2,y2= Lower right corner

          ex:[532., 299., 554., 318.]           
    IMPORTANT:        x2>x1
                      y2>y1
"""

#import cv2  
import copy
import numpy as np
import math 
from MultiObjectTracker.object import object 


class tracker:
    
    def __init__(self,names,sel_classes=None,mfc=10,max_dist=None):
        """
        The constructor accept two parameter:
            1. The class name list as input
            2. The classes to be selectively tracked : 
                default=None track all available class objects 
                or a list of index (integer) of class bject to be tracked.               
                example: [2,3,4] Then tracker only track object of class 2,3 and 4
                
            3. maximum frame count to keep missed objects (default=10 frames)
            4. max_dist: maximum distance (in pixels) of movement object centre allowed between consequtive frames to consider they are same object.
                if None: dynamic threshold will be used (the threshold will be set based on the previous object size )

        """
        if sel_classes == None:  # if selective classes not provided, then disable selective track
           self.sel_tracking=False
           self.sel_classes=None
        else:            
            self.sel_tracking=True           
            self.sel_classes=[names.index(item) for item in sel_classes]
        
        self.mfc=mfc                 
        self.names=names
        self.class_count=len(names)   # the label count
        self.fr_count=0               # Frame count to keep track of the tracking.will be incrimented on each track method call
                    
        self.old_objects={} # create old object dictionary 
        self.new_objects={} # create new object dictionary
        self.missed_objects={} # create missed object dictionary
        self.temp_objects={} # create temp object dictionary
        
        self.count=np.zeros(self.class_count,dtype=np.uint)  #create a numpy array with class size (with zero intialized) used to keep the new label for any new object for each class.
        
        if max_dist==None:           
            self.dynamic_th=True     # if max_dist is set to None then dynamic thresholding will be adapted in tracker method
            max_dist=1.00            # initialize max_dist with a dummy value
        else:
            self.dynamic_th=False    # if a valu set for max dist then dynmic threshold will be disabled and provided threshold will be used for all matching
            
        #initialize object class variable   
        #This is a threshold variable used for filtering objects those have best match probability.
        # the object class method initialize_matcher_threshold is used for this purpose
        object.initialize_matcher_threshold(max_dist)
        
    def track(self,im0,det):   
        
        object.update_curr_frame(im0) # update current frame in object class (will be used for Histogram comparison)
        self.im0=im0
        self.fr_count +=1
        """
        this method will track the object 
        Parameters
        ----------
        im0 :  numpy array
            Input image 
        det : list
            The detection output of the detector (bbox,confidence,class)
               bbox=[x1,y1,x2,y2]
                   
           x1,y1= upper left corner
           x2,y2= Lower right corner

          ex:[532., 299., 554., 318.]   
          
    IMPORTANT:        x2>x1
                      y2>y1

        Returns
        -------
        result : List (detction)
            The detetion with tracked labels
            [bbox,label(string),class]

        """
        det=det.cpu().detach().numpy().astype('float32')   # detach tensors to numpy
        self.new_objects={}                             #make new_objects empty
        self.create_temp_objects(im0,det)               #create temporary objects from new detections with unique temporary id.
        self.find_match_for_old_objects()               # find best match for all old objects from temporary objects 
        self.update_and_move_old_objects_to_new_objects()  #move old objects to new_object by updating their new location and parameters
        self.find_and_update_new_objects()              # find and update if any new object added to the scene
        
        result=self.get_label_and_bbox_for_plotiing()   # find detection for all objects in the current frame with updated label
        
        self.clean_and_update_missed_objects()          # clean missed objects
        self.old_objects=copy.deepcopy(self.new_objects)  # replace old objects with new objects for next stage
        
        return result  # return detection with new id for plotting/further application

    def create_temp_objects(self,im0,detections):
        """
        this method will create new objects from the detection
        the new objects will be created in self.temp_objects for further processing
        Parameters
        ----------
        im0 :  numpy array
            Input image 
        det : list
            The detection output of the detector (bbox,confidence,class)
               bbox=[x1,y1,x2,y2]
                   
           x1,y1= upper left corner
           x2,y2= Lower right corner

          ex:[532., 299., 554., 318.]   
          
        IMPORTANT:        x2>x1
                      y2>y1
        """
        #The new object dictionary should be cleared after each frames.
        #the following if condition take care of that 
        self.temp_objects={}

        obj_c=0
        for *xyxy, conf, cls in reversed(detections):
            
            if self.sel_tracking:                     # if selective tracking enabled           
                if cls not in self.sel_classes:       # then skip object of classes that are not part of sel_classes
                    continue
            bbox=[int(item) for item in xyxy]  #convert bbox points to ints
            obj_c+=1
            label=str(obj_c)+"_"+self.names[int(cls)]  # create label that do not conflict with old object label
            self.temp_objects.update({label:object(im0,bbox,conf,int(cls))})  # add all temp objects to temp_objects dictionary


    def find_match_for_old_objects(self):
        """
        This method identify best match for all old objects from self.temp_objects
        The match wil be saved in self.old_new_matches dictionary
        
        dict content will be as follows
         
        key:   old_object_label 
        value: [new_object_label,match_score]
            
        Returns
        -------
        None.

        """
        self.old_new_matches={}
        new_old_matches={}
        threshold=object.match_thresholds #threshold for checking the matching
        
        # find old object to new matches
        for old_key,old_obj in self.old_objects.items():  
            best_match=[]
            if self.dynamic_th:                  # only if dynamic threshold set to True (max_dist=None)
                threshold[1]=old_obj.threshold   # update neareset object seraching limit from old object parameter (close range to search)
            
            
            for new_key,new_obj in self.temp_objects.items():
                
                match_score=old_obj.get_match_score(new_obj)
                if object.is_matching(match_score,threshold):        # if old obejct and new objects matches (cla and closeness: first level filtering) 

                   if len(best_match)==0:                            # if there was no match earlier for old object
                        best_match=[new_key,match_score]

                        
                        if new_key in new_old_matches.keys():  # new key have already a match with old objects
                           """
                            If there is already a match for new_key in new_old_matches then,
                            1. identify the matched old object-1 and match_score-1
                            2. get best match of  match_score-1 with current match_score
                            3. If the match_score-1 is best then it says that old match of new_object was better
                               And new match can be ignored
                            4. if the match_score is best then it says that earlier match of new_object was wrong
                               And old match need to be removed and new match need to be updated
                           """
                           oldkey_score = new_old_matches.get(new_key)
                           res=object.get_best_match(oldkey_score[1],match_score) # compare current score and prev score
                           if res==2:  # it says new match is better than earlier match
                               self.old_new_matches.pop(oldkey_score[0])         # remove earlier match from old_new_matches
                               self.old_new_matches.update({old_key:best_match}) # add new match 
                               new_old_matches.update({new_key:[old_key,match_score]})   # update new_old_matches with new key and old_obj and match score
                           else:
                               best_match=[]                                        #else dont change history. But clear best match for next iteration
                           
                        
                        else:                                  # new key do not have a match with old objects
                            new_old_matches.update({new_key:[old_key,match_score]})
                            self.old_new_matches.update({old_key:best_match})
                        
                   else:                                          # If there was already a match then compare and get best match
                        res=object.get_best_match(best_match[1],match_score) # compare current score and prev score
                        if res==2 :                              #if best match is match_score or new_match then update the matches 

                           
                           if new_key in new_old_matches.keys():  # new key have already a match with old objects
                              """
                               If there is already a match for new_key in new_old_matches then,
                               1. identify the matched old object-1 and match_score-1
                               2. get best match of  match_score-1 with current match_score
                               3. If the match_score-1 is best then it says that old match of new_object was better
                                  And new match can be ignored
                               4. if the match_score is best then it says that earlier match of new_object was wrong
                                  And old match need to be removed and new match need to be updated
                              """
                              oldkey_score = new_old_matches.get(new_key)
                              res=object.get_best_match(oldkey_score[1],match_score) # compare current score and prev score
                              if res==2:                                        # it says new match is better than earlier match
                                  
                                  new_old_matches.pop(best_match[0])    # remove earlier entry for new_old_match by taking old Id from previous best_match
                                  best_match=[new_key,match_score]      # then update best match
                                
                                  self.old_new_matches.pop(oldkey_score[0])         # remove earlier match from old_new_matches
                                  self.old_new_matches.update({old_key:best_match})   # add new match 
                                  new_old_matches.update({new_key:[old_key,match_score]})   # update new_old_matches with new key and old_obj and match score
                              else:
                                  """
                                  Here old object (Lets say O1) have best match with the new object (lest say N2 ) compared to previous new object (lets say N2)
                                  
                                  ie:  O1-N1 match < O1-N2 match
                                  
                                  and  N2 have best match with another old object (lets say O2) compared to O1.
                                  
                                  ie: O1-N2 match< O2-N2 match
                                  
                                  
                                  We cannot pair O1 with N2, even though O1-N1 match  < O1-N2 match.
                                  Hence O1 will be paired with N1 only ie earlier best_match value will be considered as best.
                                                                 
                                  """
                                  pass                                         #else dont touch the best_match
                              
                           
                           else:                                  # new key  do not have a match with old objects
                               
                               """
                               ie:  O1-N1 match < O1-N2 match and N2 dont have any other match.
                               
                               so pair O1 with N2
                               """
                            
                               new_old_matches.pop(best_match[0])    # remove earlier entry for new_old_match by taking new Id from previous best_match
                               
                               best_match=[new_key,match_score]      # then update best_match with new matched id and score for next iteration
                               
                               new_old_matches.update({new_key:[old_key,match_score]})
                               self.old_new_matches.update({old_key:best_match})  # add new match 
                           
                           
                        else:
                            """
                            ie:  O1-N1 match > O1-N2 match.
                            
                            so no need to modify earlier match
                            Best_match will be kept as like that
                            """
                            pass                     #else dont touch the best_match
                           

                   
                   
                       
            #if len(best_match)!=0: #if there is a valid match for old object then add then to old_new_matches               
            #     self.old_new_matches.update({old_key:best_match})

        ##### after looping through all old_objects
        else:
            pass
            # if there is no items in old objects then the loop simply exit (in case of first frame too)
            # then all temp objects will be considered as new objects
            
            # if no matches from old objects
            # then all old objects will be considered as missed objects
            # then all temp objects will be considered as new objects
            pass  
     
        
     
    def update_and_move_old_objects_to_new_objects(self):       
        
        """
        This part does copying of old object to new object with matched object update
        
        if there is no match, then none of the old objects will be copied to new objects
        """

        for old_label,key_score in self.old_new_matches.items():
            """
            iterate through all items in old_new_matches
            pop old obj from old_objects
            pop matched obj from temp_objects
            update poped old object weith matched object
            move updated object to new_objects with old obj label
            """
            old_obj=self.old_objects.pop(old_label)
            matched_obj=self.temp_objects.pop(key_score[0])
            
            old_obj.update_object(matched_obj)
            
            self.new_objects.update({old_label:old_obj})
    
            
    
    def find_and_update_new_objects(self):        
        """
        Below part move remaining items in temp_objects as new items to new_objects
        """
        keys=list(self.temp_objects.keys())
        for key in keys:
            obj=self.temp_objects.pop(key)
            self.count[obj.cls]+=1
            self.new_objects.update({self.names[obj.cls]+"_"+str(self.count[obj.cls]):obj})
                



    def get_label_and_bbox_for_plotiing(self):
        """
        This method iterate through all objects in new object and return the 
        detections with updated labels
        

        Returns
        -------
        result : list
            The list contain dtection for all objects for plotiing
            one detection format looks like: [bbox,label,class]
            bbox=boundng box
            label: string label of this object 
            cls: integer value representing the class

        """
        result=[]
        for key,obj in self.new_objects.items():
            #self.im0=cv2.circle(self.im0, obj.centre, 2, (255,50,50), 2)
            result.append((obj.bbox,obj.conf,obj.cls,key))
        
        return result
        
        
        
    def clean_and_update_missed_objects(self):
        """
        This method does the following
          1. Copy un-detected objects from "old_objects" to "new_objects" if they missed only for a few frames (<mfc)
          2. Clean up the objects which are not detected upto 7 frames (mfc)
        
        """
        t_objects={}     # a new list of dictionary to track the missed objects (key=object_lable , value= how long they are missing (number of frames))

            
        for key, value in self.old_objects.items():  #iterate through all items in old_objects

            if key in self.missed_objects.keys():    #If the missed item in current frame was already part of missed objects 
                v=self.missed_objects.get(key)       # get how long it was missed (frame count)
                if v<self.mfc:
                   t_objects.update({key:v+1})    # incriment framecount for the missing object
                   
                   self.new_objects.update({key:value})        # If the object mising in less than mfc then add this object to "new_objects" , which will be further copied to old object in track method
                else:
                   self.missed_objects.pop(key)        # If the object is missing for more than mfc then remove it (this statement only help to reduce linear search of missed_object for next iteration)
                   
            else:
                t_objects.update({key:1})           #If the object is missing in current frame, then add a new entry in "temp_object"
                self.new_objects.update({key:value})   # Also copy this object to "new_object" so that it will be copied to old object in track method
                
        self.missed_objects=copy.deepcopy(t_objects)        #Update missed_object from temp_object, this will help to filter objects detected after a miss.    
                   
