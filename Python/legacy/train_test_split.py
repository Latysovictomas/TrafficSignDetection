# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:16:50 2019

@author: Work
"""

import glob
import random
import pandas as pd

    
    
    
file_list = sorted(glob.glob("dataset/FullIJCNN2013/*.ppm"))

def split(l,n):
    return random.sample(l,int(len(l)*(1-n)+0.5))

n = 0.25
train_list = split(file_list,n) 
test_list = [x for x in file_list if x not in train_list]

train_list_names = []
test_list_names = []
for path in train_list:
    name = path.split("\\")[1]
    train_list_names.append(name)
    
for path in test_list:
    name = path.split("\\")[1]
    test_list_names.append(name)
    
df_train = pd.DataFrame(train_list_names, columns = ['filename'])
df_test = pd.DataFrame(test_list_names, columns = ['filename'])

df_train.to_csv("dataset/"+"train_set.csv", sep=',', encoding='utf-8', index=False)
df_test.to_csv("dataset/"+"test_set.csv", sep=',', encoding='utf-8', index=False)
print("Success")
