import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np

def train_test_split_images(model_code=None):
    if model_code:
        if not(os.path.isdir("dataset/"+model_code)):
	        os.mkdir("dataset/"+model_code)

        full_dataset_path = "dataset/FullIJCNN2013/gt.txt"
        full_dataset_col_names = ["filename", "leftCol", "topRow", "rightCol", "bottomRow", "ClassID"]
        full_dataset_df = pd.read_csv(full_dataset_path, sep=";", header=None, names=full_dataset_col_names)

        X = full_dataset_df["filename"]
        y = full_dataset_df["ClassID"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)
    else:
        print("Cannot split, model_code not specified.")   
		
    return X_train, X_test, y_train, y_test
	
def stratifiedshufflesplit(X, y, test_size=0.2, thres=1):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, 
	


def save_to_csv_filenames(X_train, X_test, model_code=None):
    if model_code:
	    X_train.to_csv("dataset/"+model_code+"/train_set.csv", sep=',', encoding='utf-8', index=False)
	    X_test.to_csv("dataset/"+model_code+"/test_set.csv", sep=',', encoding='utf-8', index=False)
	    print("Success")
    else:
        print("Cannot save, model_code not specified.")
	
if __name__ == "__main__":
    model_code="model2"
    if not(os.path.isdir("dataset/"+model_code)):
	    os.mkdir("dataset/"+model_code)

    X_train, X_test, y_train, y_test = train_test_split_images(model_code)

    
    
	 
    save_to_csv_filenames(X_train, X_test, model_code)
	
