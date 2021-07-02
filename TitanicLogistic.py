# ===================
# Imports
# ===================
import math
import numpy as np
import pandas as pd
import seaborn as sb
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ===================
# ML Operation
# ===================
def TitanicLogistic():

    line="*"*50
    print("Logistic Regression with Titanic Dataset",line)
    
    # STEP 1 - Load Data
    titanic_Data = pd.read_csv("MarvellousTitanicDataset.csv")

    # Data Analysis
    print("First 5 record of Dataset \n")
    print(titanic_Data.head())
    
    print("Total number of records are : ",len(titanic_Data))
    print("Dataset information \n",line)
    print(titanic_Data.info())

    # STEP 2 - Analyse the data in detail
    print("\nVisualization of survived and non-survived passengers \n", line)
    figure()
    countplot(data=titanic_Data,x="Survived").set_title("Visualization according to the survived")
    show()

    figure()
    countplot(data=titanic_Data,x="Survived", hue="Sex").set_title("Visualization according to the sex")
    show()

    print("Visualization according to passenger class \n", line)
    
    # Design the graph parameter and display on the screen
    figure()
    countplot(data=titanic_Data,x="Survived", hue="Pclass").set_title("Visualization according to the PClass")
    show()


    print("survived vs non-survived based on Age")
    figure()
    titanic_Data["Age"].plot.hist().set_title("Visualization according to the Age")
    show()


    # STEP 3 - Data cleaning

    titanic_Data.drop("zero",axis=1,inplace=True)
    print("Data after coloum removel")
    print(titanic_Data.head())

    print("Gender Display")
    Sex = pd.get_dummies(titanic_Data["Sex"])
    print(Sex)
    Sex = pd.get_dummies(titanic_Data["Sex"],drop_first=True)
    print("Sex coloum after updation")
    print(Sex)
    print("Pclass Display")
    Pclass = pd.get_dummies(titanic_Data["Pclass"])
    print(Pclass)

    # concate sex and Pclass field in our Dataset
    titanic_Data = pd.concat([titanic_Data,Sex,Pclass],axis = 1)
    print("Data after concatination")
    print(titanic_Data.head())
    
    # Removing unneccessary field
    
    titanic_Data.drop(["Sex","sibsp","Embarked"],axis = 1,inplace = True)
    print(titanic_Data.head)
    
    # Divide the detaset into x and y
    x = titanic_Data.drop("Survived",axis = 1)
    y = titanic_Data["Survived"]
    
    # split the data for training and testing purpose
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.5)
    
    obj = LogisticRegression(max_iter = 2000)
    
    # STEP 4 Train the datset
    
    obj.fit(xtrain,ytrain)
    
    # STEP 5 Test the datset
    
    output = obj.predict(xtest)
    
    print("Accuracy of given dataset is : ")
    print(accuracy_score(ytest,output))
    
    print("Confusion matrix is : ")
    print(confusion_matrix(ytest,output))
# =============
# Entry Point
# =============

def main():

    print("TitanicLogistic")

    TitanicLogistic()
    print("Inside logistic function")

if __name__ == "__main__":
    main()    