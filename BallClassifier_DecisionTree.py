
from sklearn import tree
# Rough 1 # Smooth 0 # Tennis 1 # Cricket 2
def MarvellousML(weight,surface):
    # Step 1 & 2
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    Labele = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]
    # Step 3
    dobj = tree.DecisionTreeClassifier()
    #Step 4
    dobj.fit(Features,Labele)
    #Step 5
    result = dobj.predict([[weight,surface]])
    if result == 1:
        print("Your object looks like Tennis ball")
    else:
        print("Your object looks like cricket ball")
        
def main():
    print("---------- Supervised Machine Learning -----------")
    print("Enter weight of object")
    weight = int(input())
    print("Enter surface type of object")
    surface = input()
    
    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
        
    else:
        print("Error:Wrong input")
        exit()
        
    MarvellousML(weight,surface)    
        
if __name__ == "__main__":
    main()
        
