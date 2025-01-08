#!/usr/bin/env python3
# Machine Learning Project
# Authors: Bryce Strahan, Carter Bain, Ram Roy, Zack Collins


import os.path
from os import path
import urllib.request
# For python 2 use 
# import urllib
from zipfile import ZipFile
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



def download_data():
    if not path.exists('data/student-mat.csv'):             #if data is not downloaded
        print("Data not found")                 
        print("Downloading data...")
        try:
            os.mkdir('data')
            print("Directory" , '/data/' ,  "Created") 
        except FileExistsError:
            print("Directory" , '/data/' ,  "already exists")
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        urllib.request.urlretrieve(url, 'data/student.zip') #download the data (comes in as a zip)
        # for python 2 use 
        # urllib.urlopen(url)
        zfile = ZipFile('data/student.zip')
        zfile.extractall('data')                            # unzip the file to use downloaded data
    elif path.exists('data/student-mat.csv'):
        print("Data found")

def manipulate(df):
    ''' 
    ### TODO ###
    - attribute eliminations: 
        - **TODO potentially more **
    - attribute conversions:
        - **TODO potentially more **
    - 
    '''     

    ## attribute eliminations: 
    df.drop(columns=['school'], axis=1, inplace=True)
    df.drop(columns=['reason'], axis=1, inplace=True)
    df.drop(columns=['nursery'], axis=1, inplace=True)
    df.drop(columns=['Walc'], axis=1, inplace=True)
    df.drop(columns=['Dalc'], axis=1, inplace=True)

    ## conversions to binary where possible -- 
    df["sex"] = df["sex"].replace({"M": 1, "F": 0}) # female = 0,  male = 1
    df["address"] = df["address"].replace({"U": 1, "R": 0}) #rural = 0, urban = 1
    df["Pstatus"] = df["Pstatus"].replace({"T": 1, "A": 0}) # Parents together = 1, Parents apart = 0
    df["famsize"] = df["famsize"].replace({"GT3": 1, "LE3": 0}) # Greater than 3 members = 1, less than or equal to 3 members = 0
    df["schoolsup"] = df["schoolsup"].replace({"yes": 1, "no": 0}) #has extra educational school support = 1, does not have extra educational school support = 0
    df["famsup"] = df["famsup"].replace({"yes": 1, "no": 0}) # has family educationsal support = 1, does not have family educational support = 0
    df["activities"] = df["activities"].replace({"yes": 1, "no": 0}) # does extra-curricular activities = 1, does not do extra-curricular activities = 0
    df["paid"] = df["paid"].replace({"yes": 1, "no": 0}) # extra paid classes = 1, not in extra paid classes = 0
    df["internet"] = df["internet"].replace({"yes": 1, "no": 0}) # has internet access at home = 1, does not have internet access at home = 0
    df["higher"] = df["higher"].replace({"yes": 1, "no": 0}) # wants to take higher education = 1, does not want to take higher education = 0
    df["romantic"] = df["romantic"].replace({"yes": 1, "no": 0}) # in a romantic relationship = 1, not in a romantic relationship = 0 

    ## DUMMY conversions
    df = pd.get_dummies(df, columns=["Mjob"], prefix=["Mjob"], prefix_sep="-")
    df = pd.get_dummies(df, columns=["Fjob"], prefix=["Fjob"], prefix_sep="-")
    df = pd.get_dummies(df, columns=["guardian"], prefix=["guardian"], prefix_sep="-")
    return df

def multiple_linear_regression(df):
    y = df['G3'] # target value 
    df.drop(columns=['G1', 'G2', 'G3'], inplace=True) 
    x = df.values

    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state=1)

    linear_regression = LinearRegression()
    linear_regression.fit(xtrain,ytrain)
    y_pred = linear_regression.predict(x)
    



def main():
    download_data()
    df = pd.read_csv('data/student-mat.csv', sep = ';')
    df = manipulate(df)
    multiple_linear_regression(df)
    print(df)
    
if __name__ == "__main__":
    main()


