#Python3 Steven   10/08/2020 
#lxml parser:Expression profiling by array,family.xml, xml
import os
from lxml import etree

def replaceUnixEnd2Win(file_path):
    # replacement strings
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'

    with open(file_path, 'rb') as open_file:
        content = open_file.read()

    content = content.replace(UNIX_LINE_ENDING,WINDOWS_LINE_ENDING)

    with open(file_path, 'wb') as open_file:
        open_file.write(content)

class FamilyXML():
    def __init__(self,file):
        self.tree = etree.parse(file)
        self.namespace = {'x': 'http://www.ncbi.nlm.nih.gov/geo/info/MINiML' }
        
        #print(etree.tostring(self.tree.getroot()))
        print(type(self.tree))
        
        self.root = self.tree.getroot()
        print(type(self.root))
        print('items=',self.root.items()) 
        print('keys=',self.root.keys())  
        print('version=',self.root.get('version', ''))

        #for i in self.root:
        #for elem in self.tree.getiterator():
            #print(elem.tag, elem.attrib)
            
    def getSamples(self):
        def getSampleAttr(sample,name):
            xml = etree.XML(etree.tostring(sample))
            return xml.xpath("//x:" + name, namespaces=self.namespace)
        
        def getSampleAttrW(sample,name,dict):
            xml = etree.XML(etree.tostring(sample))
            return xml.xpath("//x:" + name + "[@tag='diagnosis']", namespaces=self.namespace)
        
        samples = {}
        for s in self.getChilds('Sample'):
            #res = getSampleAttr(s,'Title')
            
            if 0:
                res = getSampleAttrW(s,'Characteristics',{'tag':'diagnosis'}) #GSE114783
            else:
                res = getSampleAttr(s,'Source') #GSE25097
            
            #print(s.get('iid'), res[0].text)
            samples[s.get('iid')] = res[0].text.strip()
        
        return samples
        
    def getChilds(self,name):        
        res = self.tree.xpath("//x:" + name, namespaces=self.namespace)
        #print(res,len(res))
        return res
        # print(res[0].tag)
        # print(res[0].text)
        # print(res[0].get('iid'))
        
        # cb1 = etree.XML(etree.tostring(res[0]))
        # res = cb1.xpath('//x:' + 'Line', namespaces=namespace)
        # print('cb1=',cb1,res,len(res))
        # print(res[0].tag)
        # print(res[0].text)
        
    def parserXml(self,xmlFile):
        self.tree = etree.parse(xmlFile)
        
    
    
def main():
    file = r'.\data\GSE114783\family-xml\GSE114783_family.xml'
    #file = r'test.xml'
    #replaceUnixEnd2Win(file)
    
    xml=FamilyXML(file)
    #xml.getChilds('Sample')
    samples = xml.getSamples()
    
    for s in samples:
        print(s,samples[s])
        
if __name__=='__main__':
    main()