#python3 steven
import argparse
import numpy as np
import os

def pathsFiles(dir,filter='',subFolder=False): #"cpp h txt jpg"
    def getExtFile(file):
        return file[file.find('.')+1:]
    
    def getFmtFile(path):
        #/home/User/Desktop/file.txt    /home/User/Desktop/file     .txt
        root_ext = os.path.splitext(path) 
        return root_ext[1]

    fmts = filter.split()    
    if fmts:
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if getExtFile(getFmtFile(filename)) in fmts:
                    yield dirpath+'\\'+filename
            if not subFolder:
                break
    else:
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                yield dirpath+'\\'+filename  
            if not subFolder:
                break    
            
def createPath(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
def getFileName(path):  
    return os.path.basename(path)
    
def main():
    pass

if __name__ == '__main__':
    main()
