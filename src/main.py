#python3 Steven 
import Bio ##pip install biopython
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
#dataset refence: https://www.ncbi.nlm.nih.gov/gene/?term=Hepatitis+B+virus+genome
#tranlation: https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi
from batchPath import *
import matplotlib.pyplot as plt
from BioTest import *

def writeFasta(seqRecord,file):
    with open(file, 'w') as f_out:
        r=SeqIO.write(seqRecord, f_out, 'fasta')
        if r!=1: 
            print('Error while writing sequence:  ' + seqRecord.id)
        
def parseGenBank(file,verbose=False):
    for seq_record in SeqIO.parse(open(file, mode='r'), 'genbank'):
        # remove .id from .description record (remove all before first space)
        seq_record.description=' '.join(seq_record.description.split()[1:])
        # do something (print or edit seq_record)     
        if verbose:
            printSeqRecord(seq_record)
        #return seq_record
    
def testparser():
    file_in =r'.\data\test\testGene.txt'
    file_in =r'.\data\DQ315780.1.seq'
    file_out=r'.\data\test\testGene.fasta'
    file_gbk =r'.\data\test\testGene_genbank.txt'
    
    #help(SeqIO)
    parseFasta(file_in,True)
    #parseGenBank(file_gbk,True)
    
def getGenesStat(path):
    genomesDict = {
        'DQ315786.1':'EI057',
        'DQ315785.1':'EI0386',
        'DQ315784.1':'EI03101',
        'DQ315783.1':'EI03188',
        'DQ315782.1':'EI0423',
        'DQ315781.1':'EI0398',
        'DQ315780.1':'EI02456',
        'DQ315779.1':'EI0399',
        'DQ315778.1':'EI0388',
        'DQ315777.1':'EI03194',
        'DQ315776.1':'EI00615'}
    
    ids=[]
    lens=[]
    for i in pathsFiles(path,'txt'):
        #print(i)
        record = parseFasta(i)
        ids.append(genomesDict[record.id])
        lens.append(len(record.seq))
    return ids,lens

def plotGenesStat():
    path=r'.\data'
    ids,lens = getGenesStat(path)
    print(lens)
    
    #plt.plot(ids,lens)
    if 1:
        plt.bar(ids,lens)    
        x_offset = -0.3
        y_offset = 20.0
        w=0.99
        x = x_offset
        for l in lens:
            s = "{}".format(l)  
            y = l+y_offset
            plt.text(x, y, s, fontsize=8)
            x += w
    else:
        plt.barh(ids,lens)
        x_offset = 1
        y_offset = -0.2
        w=1.0
        y = y_offset
        for l in lens:
            s = "{}".format(l)  
            x = l+x_offset
            plt.text(x, y, s, fontsize=8)
            y += w
            
    plt.xticks(rotation=30,ha="right")
    #plt.setp(plt.get_xtickslabel(), rotation=30, ha="right",fontsize=8)
    plt.title('HBV Genome Seq Statistics')
    plt.xlabel('HBV isolate ID')
    plt.ylabel('Sequence length')
    plt.show()
    
    
def main():
    print(Bio.__version__)
    testparser()
    #plotGenesStat()
    
if __name__=='__main__':
    main()