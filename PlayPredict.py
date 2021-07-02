import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def Predictor():
    data = pd.read_csv(path)
    print("Dataset loaded succesfully with the size",len(data))
    
    Features = ["Wether","Temperature"]
    print("Fetures names are",Features)
    
    Wether = data.Wether
    Temperature = data.Temperature
    Play = data.Playn
    
    lobj = preprocessing.LabelEncoder()
    
    WetherX = lobj.fit_transform(Wether)
    TemperatureX = lobj.fit_transform(Temperature)
    Label = lobj.fit_transform(play)
    
    print("Encoded Wether is : ")
    print(WetherX)
    
    print("Encoded Temperature is : ")
    print(TemperatureX)
    
    Features = list(zip(WetherX,TemperatureX))
    
    #Step 3
    obj = KNeighborsClassifier(n_neighbors = 3
    obj.fit(Features,Label)
    
    #step 4
    output = obj.predict([[0,2]])
    
    if output == 1:
        print("you can play")
    else:
        print("Dont play")
        
    

def main():
    
    print("Enter the path of file which contains dataset")
    path = input()
    
    Predictor(path)
    
if __name__ == "__main__":
    main()