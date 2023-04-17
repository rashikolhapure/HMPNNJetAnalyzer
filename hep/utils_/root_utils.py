

import numpy as np
from ROOT import TLorentzVector
import sys
att_dict={"image":["eta","phi","pt"],"lhc":["pt","eta","phi","mass"],"lorentz":["px","py","pz","E"]}
def GetTLorentzVector(array,format="lhc",particle="visible"):
    '''Get numpy.ndarray of elements TLorentzVector from <array> of elements either a PseudoJet or 
    numpy.ndarray in format: lhc with [pt,eta,phi] or [pt,eta,phi,mass] and lorentz with [px,py,pz,E]'''
    if format=="fatjet":
        vec=np.array([TLorentzVector() for i in range(len(array))])
        count=0
        for item in array:
            vec[count].SetPtEtaPhiM(item.pt,item.eta,item.phi,item.mass)
            count+=1
        return vec
    #assert format in ("lhc","lorentz") and (array.shape[-2]==3 or array.shape[-1]==4) and len(array.shape)==2
    if len(array.shape)==1:
        vec=TLorentzVector()
        if particle=="MET": assert format=="lhc"
        if format=="lhc":
            if len(array)==4: mass=array[-1]
            else: mass=0.
            if particle=="MET":vec.SetPtEtaPhiM(array[0],0.,array[2],0.)
            else: vec.SetPtEtaPhiM(array[0],array[1],array[2],mass)
        else:
            vec.SetPxPyPzE(array[1],array[2],array[3],array[0])
        return vec
    else:
        assert particle=="visible"
        vec=[TLorentzVector() for i in range(len(array))]
        if format=="lhc":
            if array.shape[-1]==3: mass=np.zeros(len(array))
            else: mass=array[:,-1]
            for i in range(len(vec)): 
                vec[i].SetPtEtaPhiM(array[i][0],array[i][1],array[i][2],mass[i])
        elif format=="image":
            for i in range(len(vec)):
                vec[i].SetPtEtaPhiM(array[i][2],array[i][0],array[i][1],10e-16)
                Print(vec[i])
        else:
            for i in range(len(vec)): vec[i].SetPxPyPzE(array[i][0],array[i][1],array[i][2],array[i][3])
        return np.array(vec)
def GetNumpy(vectors,format="image",observable_first=True):
    '''convert TLorentzVector(s) to numpy.ndarray with format in {"image":[eta,phi,pt],"lhc":[pt,eta,phi,mass],"lorentz":[px,py,pz,E]
    default observable_first=True gives shape of (len(format),constituents) otherwise the axes are swaped''' 
    if format=="image":
        if type(vectors)==TLorentzVector: return np.array([vectors.Eta(),vectors.Phi(),vectors.Pt()],dtype="float64")
        else:
            return_array=np.zeros((len(vectors),3),dtype="float64")
            for i in range(len(vectors)):
                return_array[i]=np.array([vectors[i].Eta(),vectors[i].Phi(),vectors[i].Pt()],dtype="float64")                
    elif format=="lhc":
        if type(vectors)==TLorentzVector: return np.array([vectors.Pt(),vectors.Eta(),vectors.Phi(),vectors.M()],dtype="float64")
        else:
            return_array=np.zeros((len(vectors),4),dtype="float64")
            for i in range(len(vectors)):
                if vectors[i].M()==0 and vectors[i].P()==0: continue
                return_array[i]=np.array([vectors[i].Pt(),vectors[i].Eta(),vectors[i].Phi(),vectors[i].M()],dtype="float64")
    elif format=="lorentz":
        if type(vectors)==TLorentzVector: return np.array([vectors.Px(),vectors.Py(),vectors.Pz(),vectors.E()],dtype="float64")
        else:
            return_array=np.zeros((len(vectors),4),dtype="float64")
            for i in range(len(vectors)):
                if vectors[i].M()==0 and vectors[i].P()==0: continue
                return_array[i]=np.array([vectors[i].Px(),vectors[i].Py(),vectors[i].Pz(),vectors[i].E()],dtype="float64")
    else:
        raise ValueError("Choose format from ('image','lhc',lorentz')")
    if observable_first: return np.swapaxes(return_array,0,1)
    else: return return_array

def Sort(array,attribute="pt",order="desc"):
    '''sort a numpy.ndarray of TLorentzVectors with attribute in ("p","pt","px","py","pz","eta","phi","mass","E"). 
    descending if <order> is "desc" ortherwise ascending'''
    if attribute=="pt":indices=np.argsort([item.Pt() for item in array])
    elif attribute=="px":indices=np.argsort([item.Px() for item in array])
    elif attribute=="py":indices=np.argsort([item.Py() for item in array])
    elif attribute=="pz":indices=np.argsort([item.Pz() for item in array])
    elif attribute=="p": indices=np.argsort([item.P() for item in array])
    elif attribute=="eta":indices=np.argsort([item.Eta() for item in array])
    elif attribute=="phi":indices=np.argsort([item.Phi() for item in array])
    elif attribute=="mass": indices=np.argsort([item.M() for item in array])
    elif attribute=="E": indices=np.argsort([item.E() for item in array])
    else: raise ValueError
    if order=="desc":return np.flip(array[indices])
    else: return array[indices]

def Print(vector,format="lhc",name=None):
    '''TLorentzVector print utility function'''
    if name != None: print (name)
    if type(vector)==TLorentzVector: 
        if format=="lhc": print (f"    Eta: {vector.Eta():20.16f}        Phi: {vector.Phi():20.16f}        Pt : {vector.Pt():20.16f}        Mass: {vector.M():20.16f}        P= {vector.P():20.16f}")
        else: print (f"    Px: {vector.Px():20.16f}        Py: {vector.Py():20.16f}        Pz : {vector.Pz():20.16f}        E: {vector.E():20.16f}        P={vector.P():20.16f}")
    else:
        print ("Constituents of array: ")
        for item in vector: 
            if format=="lhc": print (f"    Eta: {item.Eta():20.16f}        Phi: {item.Phi():20.16f}        Pt : {item.Pt():20.16f}        Mass: {item.M():20.16f}        P= {item.P():20.16f}")
            else: print (f"    Px : {item.Px():20.16f}        Py: {item .Py():20.16f}        Pz: {item.Pz():20.16f}        E : {item.E():20.16f}        P= {item.P():20.16f}")
    return
def Broadcast(fatjets,check=False):
    return_array=[]
    for i in range(len(fatjets)):
        if i ==0:
            item=list(fatjets[i])
            item.append(TLorentzVector())
            item=np.array(item)
        else:item=fatjets[i]
        return_array.append(item)           
    return_array=np.array(return_array)
    if check:
        for  i in range(len(fatjets)):
            #print (fatjets[i].shape,return_array[i].shape,type(fatjets[i]),type(return_array[i]))
            s1,s2=np.sum(fatjets[i]),np.sum(return_array[i])
            assert s1.M()==s2.M() and s1.Pt() == s2.Pt() and s1.Eta() == s2.Eta() and s1.Phi()==s2.Phi()
            print (s1.M(),s2.M(),s1.Eta(),s2.Eta(),s1.Phi(),s2.Phi(),s1.Pt(),s2.Pt())
    return return_array
