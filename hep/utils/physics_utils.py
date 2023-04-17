import numpy as np
from itertools import combinations,permutations

class Legend:
    LhcIndex={0:"Eta",1:"Phi",2:"pt",3:"E"}
    Metric=np.array([1,-1,-1,-1])



def MinkowskiDot(a,b):
    assert a.shape==b.shape and a.shape[-1]==4, "No Lorentz Axis"
    InitShape=a.shape
    ReturnShape=tuple(InitShape[i] for i in range(len(InitShape)-1))
    print (InitShape,ReturnShape)
    a,b=a.reshape(-1,4),b.reshape(-1,4)
    ReturnArray=np.zeros(a.shape[0])
    for i in range(len(ReturnArray)):
        ReturnArray[i]= a[i,0]*b[i,0]-np.sum(a[i,1:]*b[i,1:])
    print (ReturnArray,"\n",ReturnArray.reshape(ReturnShape))
    return ReturnArray.reshape(ReturnShape)

def convert_to_lhc(Array):
    '''takes as input an array with last dimension 4, in [px,py,pz,E] 4vec 
    and converts it to ["pt","eta","phi","E"]
    '''
    assert Array.shape[-1] == 4, "No Lorentz Axis"
    InitShape=Array.shape
    Array=Array.reshape(-1,4)
    #print (Array)
    #Array=Array.reshape(InitShape)
    #print (Array)
    #sys.exit()
    ReturnArray=np.zeros((Array.shape))
    for i in range(len(Array)):
        ReturnArray[i]=np.array([np.sqrt(Array[i,0]**2+Array[i,1]**2),-np.log(np.tan(np.arccos(Array[i,2]/np.sqrt(Array[i,0]**2+Array[i,1]**2+Array[i,2]**2))/2.)),
                                  np.arctan2(Array[i,1],Array[i,0]),Array[i,3]])
    #print ("CONVERT",ReturnArray.reshape(InitShape),"\n")
    return ReturnArray.reshape(InitShape)

def Boost(particle,direction,eta):
    assert abs(np.sum(direction**2)-1.) <1e-12
    particle,direction=np.array(particle),np.array(direction)
    assert len(particle) == 4 and len(direction) == 3 
    E,p=particle[0],Euclid3Norm(particle[1:]*direction)
    #print (particle,E,p)
    return  np.array([E*np.cosh(eta)+p*np.sinh(eta),E*np.sinh(eta)+p*np.cosh(eta)])
def SumCombinations(FourVectors,Map=None,comb=2):
    assert len(FourVectors.shape)==2 and FourVectors.shape[1]==4, "Invalid argument as FourVectors"
    if Map==None:
        Map=list(combinations(np.arange(len(FourVectors)),comb))
        ReturnMap=True
    else:
        Map=list(Map)
        ReturnMap=False
    ReturnArray,count=np.zeros((len(Map),4),dtype="float64"),0
    for item in Map:
        ReturnArray[count]=np.sum(np.take(FourVectors,item,axis=0),axis=0)
        count+=1
    if ReturnMap:
        return ReturnArray,tuple(Map)
    else:
        return ReturnArray

def UnequalSet(*args):
    for i in range(len(args)-1):
        assert len(list(args[i]))==len(list(args[i+1])) and type(args[i])==type(args[i+1])
        for item in list(args[i]):
            assert args[i].count(item) ==1
            if item in list(args[i+1]):
                return False
    else:
        return True
def MapDict(Map):
    ReturnDict,count=dict(),0
    for i in range(len(Map)):
        for j in range(i+1,len(Map)):
            if UnequalSet(Map[i],Map[j]):
                ReturnDict["Map_"+str(count)]=[Map[i],Map[j]]
                count+=1
    return ReturnDict
def GetMass(particle):
    assert particle.shape[-1]==4 
    if len(particle.shape)==1:
        return particle[0]*np.sqrt(1-np.sum(particle[1:]**2)/particle[0]**2)
    else:
        init_shape=list(particle.shape)
        #print (particle)
        particle=particle.reshape(-1,4)
        return_array=np.zeros(particle.shape[0])
        count=0
        for item in particle:
            return_array[count]=item[0]*np.sqrt(1-np.sum(item[1:]**2)/item[0]**2)
            count+=1
        return_array=return_array.reshape(tuple(init_shape[:-1]))
        return return_array
def Get3Direction(FourVector):
    assert len(FourVector)==4 
    Dir=FourVector[1:]/Euclid3Norm(FourVector)
    assert abs(Euclid3Norm(Dir)-1)<1e-12
    return Dir
def GetEta(FourVector):
    assert len(FourVector)==4
    return np.arctanh(Euclid3Norm(FourVector)/FourVector[0])
def Euclid3Norm(FourVector):
    if len(FourVector.shape)==1 and len(FourVector)==3: 
        FourVector=np.concatenate(([0.],FourVector),axis=0)
        return np.sqrt(np.sum(FourVector[:3]**2,axis=0))
    return_array=np.zeros((len(FourVector)),dtype="float64")
    return_array=np.sum(FourVector[:3,]**2,axis=0)
    print (return_array)
    return return_array
























