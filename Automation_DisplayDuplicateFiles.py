from sys import *
import os
import hashlib

def CalculateChecksum(path, blocksize = 1024):
    fd = open(path,'rb')
    hobj = hashlib.md5()
    # 2051
    buffer = fd.read(blocksize)
    while len(buffer) > 0:
        hobj.update(buffer)
        buffer = fd.read(blocksize)

    fd.close()
    return hobj.hexdigest()
    
def DirectoryTraversal(path):
    print("Contents of the directory are")
    
    duplicate = {}       # Dictonary to hold checksum and file name
    for Folder , Subfolder , Filename in os.walk(path):
        print("Directory name is : "+Folder)
        for sub in Subfolder:
            print("Subfolder of  " + Folder + "is "+ sub)
        for file in Filename:
            print("File name is : "+file)
            actualpath = os.path.join(Folder,file)
            hash = CalculateChecksum(actualpath)
            
            if hash in duplicate:       #that checksum is already present
                duplicate[hash].append(actualpath)
            else:                           # there is no such checksum
                duplicate[hash] = [actualpath]
            
    return duplicate
 

def DisplayDuplicate(duplicate):
    output = list(filter(lambda x : len(x) > 1, duplicate.values()))
    
    if(len(output) > 0):    # 2
        print("There are duplicate files")
    else:
        print("There are no dupicate files")
        return
    
    print("List of duplicate files are : ")
    
    i = 0
    icnt = 0
    for result in output:
        icnt = 0
        for path in result:
            icnt+=1
            if icnt >=2:
                i +=1
                os.remove(path)
                
    print("Number of duplicate files deleted : ",i)
    
def main():
    print("Marvellous Infosystems")
    print("Directory travesal script")
    
    if(len(argv) != 2):
        print("Error : Invalid number of arguments")
        exit()
        
    if(argv[1] == "-h") or  (argv[1] == "-H"):
        print("It is a Directory cleaner script")
        exit()
        
    if(argv[1] == "-u") or (argv[1] == "-U"):
        print("Usage : Provide the absolute path of the target director")
        exit()

    arr = {}
    arr = DirectoryTraversal(argv[1])
    
    DisplayDuplicate(arr)
    
if __name__ == "__main__":
    main()