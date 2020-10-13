#python3 Steven 
import Bio
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Seq import MutableSeq
from Bio.Seq import UnknownSeq
from Bio.Seq import complement,reverse_complement, transcribe, back_transcribe, translate
from Bio import AlignIO

from Bio.Alphabet import IUPAC
from Bio.SeqUtils import GC
from Bio.Alphabet import generic_alphabet
from Bio.Alphabet import generic_dna,generic_protein
from Bio.Data import CodonTable
from Bio.Blast import NCBIWWW
from Bio.Blast.Applications import NcbiblastxCommandline
import numpy as np


def testAlphabets():
    #my_seq = Seq("AGTACACTGGT")
    #my_seq = Seq("AGTACACTGGT", IUPAC.unambiguous_dna)
    my_seq = Seq("AGTACACTGGT", IUPAC.protein)
    print('My_Seq = ', my_seq)
    print('alphabet = ', my_seq.alphabet)
    print('Seq len = ', len(my_seq))
    #for index, letter in enumerate(my_seq):
    #    print("%i %s" % (index, letter))
    
    uniq = uniue_char = list(set(str(my_seq)))
    print('uniq=',uniq)
    for i in uniq:
        print(i,'count=',my_seq.count(i))
        
    print(100 * float(my_seq.count("G") + my_seq.count("C")) / len(my_seq))
    print('GC percentage=',GC(my_seq))
    
    newSeq = my_seq[4:9]
    print('newSeq = ', newSeq)
    
    print('reverse = ',my_seq[::-1]) #reverse seq
    
def ConcatenatingSeq():
    protein_seq = Seq("EVRNAK", IUPAC.protein)
    dna_seq = Seq("ACGT", IUPAC.unambiguous_dna)
    
    #print(protein_seq + dna_seq) #error 
    protein_seq.alphabet = generic_alphabet
    dna_seq.alphabet = generic_alphabet
    print(protein_seq + dna_seq)
    
    list_of_seqs = [Seq("ACGT", generic_dna), Seq("AACC", generic_dna), Seq("GGTT", generic_dna)]
    concatenated = Seq("", generic_dna)
    for s in list_of_seqs:
        concatenated +=s
    print('concatenated=',concatenated)

    con = sum(list_of_seqs, Seq("", generic_dna))
    print('con=',con)
    
    dna_seq = Seq("acgtACGT", generic_dna)
    print('unper=',dna_seq.upper())
    print('lower=',dna_seq.lower())
    print("GTAC" in dna_seq)
    print("GTAC" in dna_seq.upper())
    
def complementSeq():
    my_seq = Seq("GGATCGAAATCGC", IUPAC.unambiguous_dna)
    print('My_Seq = ', my_seq)
    print('My_Seq complement = ', my_seq.complement())
    print('My_Seq reverse complement = ', my_seq.reverse_complement())
    print('My_Seq reverse reverse complement = ', my_seq.reverse_complement().reverse_complement())
    
    #protein_seq = Seq("EVRNAK", IUPAC.protein)
    #print(protein_seq.complement()) #error Proteins do not have complements!
    
def transcriptionSeq():
    coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", IUPAC.unambiguous_dna)
    complement_dna = coding_dna.complement()
    print('coding_dna = ', coding_dna)
    print('complement_dna = ', complement_dna)
    
    messenger_rna = coding_dna.transcribe()
    print('messenger_rna = ', messenger_rna)
    
    back = messenger_rna.back_transcribe()
    print('back = ', back)
    '''
    coding_dna =  ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG
    complement_dna =  TACCGGTAACATTACCCGGCGACTTTCCCACGGGCTATC
    messenger_rna =  AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG
    back =  ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG
    '''
    
def translationCOdingDNA(dnaSeq,tableId):
    table = CodonTable.unambiguous_dna_by_id[tableId] # 1~6
    #dnaSeq = Seq(dnaSeq, IUPAC.unambiguous_dna) #string
    proteinD = dnaSeq.translate(table=table)
    print('tableId = ',tableId,'proteinD = ', proteinD)
    #print('table = ',table,'\nproteinD = ', proteinD)
    return proteinD

def translationSeq():
    coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", IUPAC.unambiguous_dna)
    if 1:
        for i in range(1,7):
            translationCOdingDNA(coding_dna,i)
    else:
        messenger_rna = coding_dna.transcribe()
        protein = messenger_rna.translate()
        print('messenger_rna = ', messenger_rna)
        print('protein = ', protein)
        
        proteinD = coding_dna.translate()
        print('proteinD = ', proteinD)
        
        proteinC = coding_dna.translate(table="Vertebrate Mitochondrial")
        print('proteinC = ', proteinC)
        proteinC = coding_dna.translate(table=2)
        print('proteinC = ', proteinC)
        proteinC = coding_dna.translate(table=2,to_stop=True)
        print('proteinC = ', proteinC)
        proteinC = coding_dna.translate(table="Bacterial")
        print('proteinC = ', proteinC)
        '''
        messenger_rna =  AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG
        protein =  MAIVMGR*KGAR*
        proteinD =  MAIVMGR*KGAR*
        proteinC =  MAIVMGRWKGAR*
        proteinC =  MAIVMGRWKGAR*
        proteinC =  MAIVMGRWKGAR
        proteinC =  MAIVMGR*KGAR*
        '''
    
def TranslationTables():
    standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
    mito_table = CodonTable.unambiguous_dna_by_name["Vertebrate Mitochondrial"]
    
    # standard_table = CodonTable.unambiguous_dna_by_id[1]
    # mito_table = CodonTable.unambiguous_dna_by_id[2]
    
    # print('standard_table=\n',standard_table)
    # print('mito_table=\n',mito_table)

    # print('mito_table.start_codons=',mito_table.start_codons)
    # print('mito_table.stop_codons=',mito_table.stop_codons)
    
    for i in range(1,6):
        print('-------------------------------',i,'----')
        table = CodonTable.unambiguous_dna_by_id[i]
        print(table)
        
def compareSeq():
    seq1 = Seq("ACGT", IUPAC.unambiguous_dna)
    seq2 = Seq("ACGT", IUPAC.ambiguous_dna)
    print(str(seq1) == str(seq2))
    print(seq1 == seq2)
    
    dna_seq = Seq("ACGT", generic_dna)
    prot_seq = Seq("ACGT", generic_protein)
    print(dna_seq == prot_seq)
   
def MutableGeneSeq():
    my_seq = Seq("GCCATTGTAATGGGCCGCTGAAAGGGTGCCCGA", IUPAC.unambiguous_dna)
    #my_seq[5] = "G" #error 'Seq' object does not support item assignment
    
    mutable_seq = my_seq.tomutable()
    mutable_seq[1]='T'
    print('mutable_seq = ', mutable_seq)
    
    mutable_seq = MutableSeq("GCCATTGTAATGGGCCGCTGAAAGGGTGCCCGA", IUPAC.unambiguous_dna)
    mutable_seq[0]='T'
    print('mutable_seq = ', mutable_seq)
    
    new_seq = mutable_seq.toseq() #convert to readonly seq

def unknownSeq():
    #unk = UnknownSeq(20,character='*')
    unk = UnknownSeq(20, alphabet=IUPAC.ambiguous_dna)
    print('unk = ', unk)
    
def directStringSeq():
    my_string = "GCTGTTATGGGTCGTTGGAAGGGTGGTCGTGCTGCTGGTTAG"
    Compl = complement(my_string)
    reCompl = reverse_complement(my_string)
    transc = transcribe(my_string)
    bTransc = back_transcribe(my_string)
    transl = translate(my_string)
    print('my_string = ', my_string)
    print('Compl = ', Compl)
    print('reCompl = ', reCompl)
    print('transc = ', transc)
    print('bTransc = ', bTransc)
    print('transl = ', transl)

def formatFASTA():
    record = SeqRecord(
        Seq(
        "MMYQQGCFAGGTVLRLAKDLAENNRGARVLVVCSEITAVTFRGPSETHLDSMVGQALFGD"
        "GAGAVIVGSDPDLSVERPLYELVWTGATLLPDSEGAIDGHLREVGLTFHLLKDVPGLISK"
        "NIEKSLKEAFTPLGISDWNSTFWIAHPGGPAILDQVEAKLGLKEEKMRATREVLSEYGNM"
        "SSAC",
        generic_protein,
        ),
        id="gi|14150838|gb|AAK54648.1|AF376133_1",
        description="chalcone synthase [Cucumis sativus]",
        )
    print(record.format("fasta"))

def alignSeq():
    file_in = r'.\data\test\84_85.txt' #r'.\data\test\testGene.txt' #ABD36984.1 x ABD36985.1.aln
    #file_in = r'.\data\test\PF05356_seed.txt'
    file_out = r'.\data\test\testGene.fasta'
    file_gbk = r'.\data\test\testGene_genbank.txt'
    
    #help(AlignIO)
    alignment = AlignIO.read(file_in, "fasta")
    print(alignment)
    
    print("Alignment length %i" % alignment.get_alignment_length())
    for record in alignment:
        #print("%s - %s" % (record.seq, record.id))
        print(record)
        
def blastNCBI(strSeq,outFile):
    #help(NCBIWWW.qblast)  blastn: nucleotide   blastp:protein
    result_handle = NCBIWWW.qblast("blastn", "nt", strSeq)
    #print('result_handle=\n',result_handle.read())
    
    with open(outFile, "w") as out_handle:
        out_handle.write(result_handle.read())
    result_handle.close()
    
    # file_in = r'.\data\test\testGene.txt'
    # fasta_string = open(file_in).read()
    # result_handle = NCBIWWW.qblast("blastn", "nt", fasta_string)
    
def blastNCBICmd():
    file_in = r'.\data\test\testGene.txt'
    #help(NcbiblastxCommandline)
    blastx_cline = NcbiblastxCommandline(query=file_in, db="nr", evalue=0.001,
        outfmt=5, out="testGene.xml")
    print(blastx_cline)
    
def LongestCommonSubsequence(x,y):
    commonIndex=[]
    c = np.zeros((len(x)+1,len(y)+1),dtype=np.uint8)
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            if x[i] == y[j]:
                c[i+1,j+1] = c[i,j] + 1
                commonIndex.append([i,j])
            else:
                c[i+1,j+1] = np.max([c[i+1,j],c[i,j+1]])
    print('c=\n',c)
    print('res=',c[-1,-1])
    return c[-1,-1],commonIndex

def printSeqRecord(seq_record):
    print('SequenceID = '  + seq_record.id)
    print('Name = '  + seq_record.name)
    print('Letter_annotations = ', seq_record.letter_annotations)
    print('Annotations = ', seq_record.annotations)
    print('Features = ', seq_record.features)
    print('------------------------------------------------')
    for i,feature in enumerate(seq_record.features):
        print('Feature',i,'\n',feature)
    print('------------------------------------------------')
    print('Dbxrefs = ', seq_record.dbxrefs)
    
    print('Description = ' + seq_record.description + '\n')
    #print(type(seq_record.seq))
    #print(dir(Bio.Seq.Seq))
    print('Seq Len = ',len(seq_record.seq))
    print('Seq = ' + seq_record.seq,'\n')
    
def parseFasta(file,verbose=False):
    print('FASTA File:',file)
    for seq_record in SeqIO.parse(open(file, mode='r'), 'fasta'):
        # remove .id from .description record (remove all before first space)
        seq_record.description=' '.join(seq_record.description.split()[1:])
        if verbose:
            printSeqRecord(seq_record)
        return seq_record

def LCSSequence():
    f1 = r'.\data\ABD36984.1.fa'
    f2 = r'.\data\ABD36985.1.fa'
    record1 = parseFasta(f1)
    record2 = parseFasta(f2)
    
    len,indexs=LongestCommonSubsequence('aqqqqqqqqqxbc','eaex')
    print(len,indexs)
    
    #len,indexs=LongestCommonSubsequence(str(record1.seq),str(record2.seq))
    #print(len,indexs)
    
def main():
    #print(Bio.__version__)
    #testAlphabets()
    #ConcatenatingSeq()
    #complementSeq()
    #transcriptionSeq()
    translationSeq()
    #TranslationTables()
    #compareSeq()
    #MutableGeneSeq()
    #unknownSeq()
    #directStringSeq()
    #formatFASTA()
    #alignSeq()
    #blastNCBI('ATGCTG',r'.\data\test\test.xml')
    #LCSSequence()
    
if __name__=='__main__':
    main()