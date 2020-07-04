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

def toPercentage(img_orig, x1,y1,x2,y2,l):
    h,w,c = img_orig.shape;
    x1p = x1 / w;
    x2p = x2 / w;
    y1p = y1 / h;
    y2p = y2 / h;
    return x1p,y1p,x2p,y2p,l

def toImCoord(img_resized, x1p,y1p,x2p,y2p,l):
#function x1,y1,x2,y2 = toImCoord(img_resized, x1p,y1p,x2p,y2p)
    h,w,c = img_resized.shape;
    x1 = x1p * w;
    x2 = x2p * w;
    y1 = y1p * h;
    y2 = y2p * h;
    return x1,y1,x2,y2,l

mainFolder = "dataset/FullIJCNN2013/"
with open(mainFolder + "gt.txt", 'r') as f:
    file = f.readlines()
    file = [value.strip().split(';')  for value in file]  #file.split(";")
    df = pd.DataFrame(file)


df_copy = df.copy()
def read_coord(dataf, name):
    row = dataf.loc[dataf[0] == name].head(1)
    #print(row[1])
    x1 = int(row[1])
    y1 = int(row[2])
    x2 = int(row[3])
    y2 = int(row[4])
    l = int(row[5])
    #df.loc[df[0] == name].head(1).drop(axis = 0,inplace = True)
    return x1,y1,x2,y2,l
    

# Convert to percentages
print("Converting bboxes to percentages...")

perc_list = []
#file_list = sorted(glob.glob("dataset/FullIJCNN2013/*.ppm"))
for idx, file_name in enumerate(file):
    #print(file_name)
    #im = Image.open("dataset/FullIJCNN2013/"+file_name[0])
    im = cv2.imread(mainFolder + file_name[0])
    name = file_name[0]#file_name.split("2013/")[1]
    #print(df_copy.loc[df_copy[0] == name].head(1))
    x1,y1,x2,y2, l = read_coord(df_copy, name)
    df_copy.drop(df_copy.loc[df_copy[0] == name].head(1).index,inplace=True)
    x1p,y1p,x2p,y2p,l = toPercentage(im, x1,y1,x2,y2,l)
    perc_list.append([name,x1p,y1p,x2p,y2p,l])

df_perc = pd.DataFrame(perc_list)


# Resize images
print("Resizing images...")

class_name = "traffic_scene"
new_width = 300
new_height = 300

if not(os.path.isdir("dataset/images")):
    os.mkdir("dataset/images")
file_list = sorted(glob.glob("dataset/FullIJCNN2013/*.ppm"))

if len(os.listdir('dataset/images/') ) == 0:
    for idx, file_name in enumerate(file_list):
        im = Image.open(file_name)
        #new_width = new_width#640
        #new_height = new_height#400
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        name_ppm = file_name.split("\\")[1]
        name = name_ppm.split(".")[0]
        #im.save("dataset/images/" + class_name + "_" + str(idx+1).zfill(3) + ".jpg")
        im.save("dataset/images/" + name + ".jpg")

# Convert bboxes to resized image
print("Creating new bboxes for resized images...")        

bbox_list = []   
for index, row in df_perc.iterrows():
    name_ppm = row[0]
    name_resized = name_ppm.split(".")[0]
    name_resized = str(name_resized) + ".jpg"
    x1p,y1p,x2p,y2p,l = row[1], row[2], row[3], row[4], row[5]
    im = cv2.imread("dataset/images/" + name_resized) 
    x1,y1,x2,y2,l = toImCoord(im, x1p,y1p,x2p,y2p,l)
    bbox_list.append([name_resized,new_width,new_height,l,x1,y1,x2,y2])

df_bbox = pd.DataFrame(bbox_list)


# Train test split

if not(os.path.isdir("dataset/train")):
    os.mkdir("dataset/train")
if not(os.path.isdir("dataset/test")):
    os.mkdir("dataset/test")
    
file_list = sorted(glob.glob("dataset/images/*.jpg"))

def split(l,n):
    return random.sample(l,int(len(l)*(1-n)+0.5))

n = 0.25
train_list = split(file_list,n) 
test_list = [x for x in file_list if x not in train_list]

# Uncomment below to read labels from csv

#test_list_df = pd.read_csv("dataset/test_set.csv")
#train_list_df = pd.read_csv("dataset/train_set.csv")
#test_list_df["filename"] = test_list_df["filename"].str.replace('.ppm','.jpg')
#train_list_df["filename"] = train_list_df["filename"].str.replace('.ppm','.jpg')
#train_list_names = train_list_df["filename"].tolist()
#test_list_names = test_list_df["filename"].tolist()

train_list_names = []
test_list_names = []
for path in train_list:
    name = path.split("\\")[1]
    train_list_names.append(name)
    
for path in test_list:
    name = path.split("\\")[1]
    test_list_names.append(name)
    
df_test = df_bbox.loc[df_bbox[0].isin(test_list_names)]
df_train = df_bbox.loc[df_bbox[0].isin(train_list_names)]

df_test = df_test.rename({0: 'filename', 1: 'width', 2: 'height',\
                          3: 'class', 4: 'xmin', 5: 'ymin', 6: 'xmax', 7: 'ymax'}, axis=1)
df_train = df_train.rename({0: 'filename', 1: 'width', 2: 'height',\
                          3: 'class', 4: 'xmin', 5: 'ymin', 6: 'xmax', 7: 'ymax'}, axis=1)  

for idx, file_name in enumerate(test_list):
    im = Image.open(file_name)
    #im.save("dataset/test/" + class_name + "_" + str(idx+1).zfill(3) + ".jpg")
    name_ppm = file_name.split("\\")[1]
    name = name_ppm.split(".")[0]
    im.save("dataset/test/"+ name + ".jpg")
    
for idx, file_name in enumerate(train_list):
    im = Image.open(file_name)
    #im.save("dataset/train/" + class_name + "_" + str(idx+1).zfill(3) + ".jpg")
    name_ppm = file_name.split("\\")[1]
    name = name_ppm.split(".")[0]
    im.save("dataset/train/"+ name + ".jpg")
print("Splitted into train and test dataset")



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

df_test.replace({"class": d}).to_csv("dataset/"+"test_labels.csv", sep=',', encoding='utf-8', index=False)
df_train.replace({"class": d}).to_csv("dataset/"+"train_labels.csv", sep=',', encoding='utf-8', index=False)
print("Successfully created csv label files")
print(df_test["filename"].nunique())
print(df_train["filename"].nunique())


# Check if coordinates right
#files = sorted(glob.glob("../300/test/*.jpg"))
#files = sorted(glob.glob("../300/test/*.jpg"))
#dd = pd.read_csv("../300/images/")
