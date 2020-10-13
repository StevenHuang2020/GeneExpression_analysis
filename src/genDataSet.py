#python3 Steven 
import argparse
from batchPath import *
import pandas as pd
from sampleXML import FamilyXML

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help = 'path')
    #parser.add_argument('-lr', '--lr', help='learning rate')
    #parser.add_argument('-new', dest='newModel', action='store_true')
    return parser.parse_args()

def readGSMFile(file):
    df = pd.read_table(file,header=None) #read_table
    #df.set_index('0')
    df = df.T
    columns=df.iloc[0]
    df = df[1:]
    df = df.rename(columns=columns)
    
    #print('index=',df.index)
    #print('columns=',df.columns)
    #print(df)
    return df
    
def writeToCsv(df,file,index=True, header=True, sep=','): #','
    df.to_csv(file, index=index, header=header, sep=sep) 

def getData(path,dstPath,samples):
    df = pd.DataFrame()
    for i in pathsFiles(path,filter='txt'):
        fileName = getFileName(i)
        if not fileName.startswith('GSM'):
            continue
        
        gsmId = fileName[:fileName.find('-')]
        print(i,fileName,gsmId)
        line = readGSMFile(i)
        line.insert(0, 'GSM_ID', gsmId)
        
        #line.insert(1, "Type", [samples[gsmId]]) #insert special pos
        line["Type"] = samples[gsmId]             #insert to the tail
        
        # id = pd.DataFrame([['GSM_ID',gsmId]])
        # line = pd.concat([id, line])
        # line = line.append({0: 'GSM_ID', 1: gsmId}, ignore_index=True)  
        df = df.append(line)
    
    #df.set_index(['GSM_ID'], inplace=True)
    dstTxtName = dstPath[dstPath.rfind('\\')+1:] + '.txt' 
    dstName = dstPath[dstPath.rfind('\\')+1:] + '.csv'
    print(dstName)
    print(df.head())
    print(df.shape)
    
    dstTxt = dstPath + '\\' + dstTxtName
    dst = dstPath + '\\' + dstName
    writeToCsv(df, dstTxt, sep=' ', index=None, header=None)
    #writeToCsv(df, dst)
    writeToCsv(df.T, dst, header=None)
    
def genCSVData_GSE114783():
    path = r'.\data\GSE114783\family-xml' #   GSE5975 GSE52750
    dstPath = r'.\data\GSE114783'
    xml = r'.\data\GSE114783\family-xml\GSE114783_family.xml'
    
    samples = FamilyXML(xml).getSamples()
    for s in samples:
        print(s,samples[s])

    getData(path,dstPath,samples)
    
def genCSVData_GSE25097():
    path = r'.\data\GSE25097\family-xml' #   GSE5975 GSE52750
    dstPath = r'.\data\GSE25097'
    xml = r'.\data\GSE25097\family-xml\GSE25097_family.xml'
    
    samples = FamilyXML(xml).getSamples()
    for s in samples:
        print(s,samples[s])

    getData(path,dstPath,samples)
    
def main():
    #arg = argCmdParse()
    # if arg.path:
    #     path = arg.path
    #print('path=',path)
    
    #genCSVData_GSE114783()
    genCSVData_GSE25097()
    

if __name__=='__main__':
    main()
    