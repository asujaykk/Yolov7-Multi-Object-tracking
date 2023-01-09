#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 02:28:54 2022

@author: akhil_kk
"""
import numpy as np
from object import object

threshold=np.array([2.00,1.0])

match1=np.array([2.0,1.0])
match2=np.array([1.0,0.5])

x0y0=(678,284)
x1y1=()

print(object.get_closeness(x0y0, x1y1))

print(object.is_matching(match2, threshold))
print(object.get_best_match(match2, match1))