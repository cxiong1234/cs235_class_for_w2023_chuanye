#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:45:53 2022

@author: cxiong
"""
import statistics

f1_score_list = [0.6190476190476191, 0.6227544910179641, 0.6799999999999999, 0.7796610169491525, 0.525]



print("SD=", statistics.stdev(f1_score_list))

average = sum(f1_score_list) / len(f1_score_list)

print("average=",average)


