# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:20:16 2019

@author: Work
"""

import glob
from PIL import Image
import os
import pandas as pd
import cv2
import random
pd.options.mode.chained_assignment = None  # default='warn'
model_code = "model2"
if not(os.path.isdir("dataset/"+model_code)):
    os.mkdir("dataset/"+model_code)

mainFolder = "dataset/FullIJCNN2013/"
with open(mainFolder + "gt.txt", 'r') as f:
    file = f.readlines()
    file = [value.strip().split(';')  for value in file]
    df = pd.DataFrame(file)


df_copy = df.copy()
df_copy["height"] = 800
df_copy["width"] = 1360
cols = df_copy.columns.tolist()
print(cols)
df_copy = df_copy[[0, "width", 'height', 5, 1,2,3,4]]
df_copy = df_copy.rename({0: 'filename',\
                          5: 'class', 1: 'xmin', 2: 'ymin', 3: 'xmax', 4: 'ymax'}, axis=1)
print(df_copy.head())






# Train test split

if not(os.path.isdir("dataset/"+model_code+"/train")):
    os.mkdir("dataset/"+model_code+"/train")
if not(os.path.isdir("dataset/"+model_code+"/test")):
    os.mkdir("dataset/"+model_code+"/test")
    
#file_list = sorted(glob.glob("dataset/"+model_code+"/images/*.png"))

test_list_df = pd.read_csv("dataset/"+model_code+"/test_set.csv")
train_list_df = pd.read_csv("dataset/"+model_code+"/train_set.csv")

train_list_names = train_list_df["filename"].tolist()
test_list_names = test_list_df["filename"].tolist()





    






df_test = df_copy.loc[df_copy["filename"].isin(test_list_names)]
df_train = df_copy.loc[df_copy["filename"].isin(train_list_names)]

#df_test = df_test.rename({0: 'filename',\
#                          5: 'class', 1: 'xmin', 2: 'ymin', 3: 'xmax', 4: 'ymax'}, axis=1)
#df_train = df_train.rename({0: 'filename',\
#                          5: 'class', 1: 'xmin', 2: 'ymin', 3: 'xmax', 4: 'ymax'}, axis=1)

#
# Replace label to sign names and save to csv
d = {0 : "speed limit 20",
1 : "speed limit 30",
2 : "speed limit 50",
3 : "speed limit 60",
4 : "speed limit 70",
5 : "speed limit 80",
6 : "restriction ends 80",
7 : "speed limit 100",
8 : "speed limit 120",
9 : "no overtaking",
10 : "no overtaking trucks",
11 : "priority at next intersection",
12 : "priority road",
13 : "give way",
14 : "stop",
15 : "no traffic both ways",
16 : "no trucks",
17 : "no entry",
18 : "danger",
19 : "bend left",
20 : "bend right",
21 : "bend",
22 : "uneven road",
23 : "slippery road",
24 : "road narrows",
25 : "construction",
26 : "traffic signal",
27 : "pedestrian crossing",
28 : "school crossing",
29 : "cycles crossing",
30 : "snow",
31 : "animals",
32 : "restriction ends",
33 : "go right",
34 : "go left",
35 : "go straight",
36 : "go right or straight",
37 : "go left or straight",
38 : "keep right",
39 : "keep left",
40 : "roundabout",
41 : "restriction ends overtaking",
42 : "restriction ends overtaking trucks"}

print(df_test.dtypes)
df_test["class"] = df_test["class"].astype(int)
df_train["class"] = df_train["class"].astype(int)

df_test.replace({"class": d}, inplace = True)
df_train.replace({"class": d}, inplace = True)
df_test["filename"] = df_test["filename"].str.replace('.ppm','.png')
df_train["filename"] = df_train["filename"].str.replace('.ppm','.png')
                 
df_test.to_csv("dataset/"+model_code+"/test_labels.csv", sep=',', encoding='utf-8', index=False)
df_train.to_csv("dataset/"+model_code+"/train_labels.csv", sep=',', encoding='utf-8', index=False)
print("Successfully created csv label files")



# Convert to PNG format
"""
print("Saving images to png...")
if not(os.path.isdir("dataset/"+model_code+"/images")):
    os.mkdir("dataset/"+model_code+"/images")
file_list = sorted(glob.glob("dataset/FullIJCNN2013/*.ppm"))

if len(os.listdir("dataset/"+model_code+"/images") ) == 0:
    for idx, file_name in enumerate(file_list):
        im = Image.open(file_name)
        name_ppm = file_name.split("\\")[1]
        name = name_ppm.split(".")[0]
        im.save("dataset/"+model_code+"/images/" + name + ".png")
"""

split = False
if split: 
		
	# Splitting actual images
    print("Splitting actual images...")  
    for idx, file_name in enumerate(test_list_names):
        name = file_name.split(".")[0]
        im = Image.open("dataset/"+model_code+"/images/"+file_name)
        im.save("dataset/"+model_code+"/test/"+ name + ".png")
		
    for idx, file_name in enumerate(train_list_names):
        name = file_name.split(".")[0]
        im = Image.open("dataset/"+model_code+"/images/"+file_name)
        im.save("dataset/"+model_code+"/train/"+ name + ".png")



# Check if coordinates right


