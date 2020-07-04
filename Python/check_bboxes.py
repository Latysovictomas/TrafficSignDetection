# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:52:22 2019

@author: Work
"""
import glob
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
# from timeit import default_timer as timer


model_code="model"
ext = ".jpg"
df_train = pd.read_csv("dataset/"+model_code+"/train_labels.csv")


file_list_test = sorted(glob.glob("dataset/"+model_code+"/train/*"+ext))
for path in file_list_test:
    print(path)
    name = path.split("\\")[1]
    
    

    img = cv2.imread(path)
    results = df_train.loc[df_train["filename"]==str(name),["xmin", "ymin", "xmax", "ymax"]].values.tolist()

        
    rect = np.array([[x1,y1,x2,y2] for (x1,y1,x2,y2) in results])
    pick = non_max_suppression(rect, probs = None, overlapThresh = 0.2)
        
        
    for x1,y1,x2,y2 in pick:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)


    # cv named Window
    img = cv2.resize(img, (600,353))
    cv2.imshow("results", img) # cv2.resize(img, (448,448)))
    print("results", rect)
    k = cv2.waitKey(30) & 0xff
    if k ==27:
        cv2.waitKey(0)
        break
    if (len(results)>0):
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
