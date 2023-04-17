
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from hep_ml.io.saver import Unpickle,RunIO
from hep_ml.genutils import pool_splitter


def load_data(classes,length=None,suffix=None,test_train_split=0.25,input_keys=["high_level"],return_array=False,function=None,run_io=False,**kwargs):
    count=0
    X=[[] for _ in input_keys]
    for item in classes:
        if not run_io:
            if function is None:
                if "bin_name" in kwargs: folder="/"+kwargs.get("bin_name")
                else: folder="/all"
                events=Unpickle(item+".h",load_path="./processed_events/"+suffix+folder)
            else: events=pool_splitter(function,Unpickle(item+".h",load_path="./temp_data"))
        else:
            r=RunIO(item,kwargs.get("data_tag"),mode="r")
            events=r.load_events()
        for i,input_key in enumerate(input_keys):
            if input_key!="high_level":
                X[i]=np.expand_dims(events[input_key][:length],-1)
                if kwargs.get("log",False):
                    print ("Calculating log of "+input_key+"...",np.min(X[i][np.where(X[i])]),np.max(X[i][np.where(X[i])]))
                    X[i][np.where(X[i])]=np.log(X[i][np.where(X[i])])
                    print ("New: ",np.min(X[i][np.where(X[i])]),np.max(X[i][np.where(X[i])]))
            else:
                X[i]=events[input_key][:length]
        Y=np.zeros((len(X[0]),len(classes)))
        Y[:,count]=1.
        print (type(X),Y.shape)
        train_index=int(len(X)*(1-test_train_split))
        if count==0:
            X_all,Y_all=[item[:] for item in X],Y[:]
        else:
            X_all,Y_all=[np.concatenate((prev_item,item[:]),axis=0) for prev_item,item in zip(X_all,X)],np.concatenate((Y_all,Y[:]),axis=0)
        print (item,Y[-10:],len(X))
        count+=1
    if len(input_keys)==1: 
        X_all=X_all[0]
        assert X_all.shape[0]==Y_all.shape[0]
        X_train,X_val,Y_train,Y_val=train_test_split(X_all,Y_all,shuffle=True,random_state=12,test_size=0.25)
    else:
        x_length=len(X_all)
        combined=X_all+[]
        combined.append(Y_all)
        if "debug" in sys.argv:print ("combined:",combined[-1][:10],combined[-1][10:])
        combined=list(train_test_split(*combined,shuffle=True,random_state=12,test_size=0.25))
        X_train,X_val=[],[]
        for i in range(len(combined)-2): 
            print (type(combined[i]),combined[i].shape)
            if i%2==0: X_train.append(combined[i])
            else: X_val.append(combined[i])
        Y_train=combined[-2]
        Y_val=combined[-1]
    #if "debug" in sys.argv: 
    shape_print(X_train,Y_train),shape_print(X_val,Y_val)
    train_dict={"X":X_train,"Y":Y_train}
    test_dict={"X":X_val,"Y":Y_val}
    if return_array: return X_train,Y_train,X_val,Y_val
    else: return {"train":train_dict,"val":test_dict,"classes":classes}
    
def shape_print(X,Y):
    if type(X)==np.ndarray: print ("X:",X.shape)
    else: [print ("\nX"+str(i)+" :",item.shape) for i,item in enumerate(X)]
    print ("Y: ",Y.shape,"\nY head: ",Y[:5],"\nY tail:",Y[-5:])
    return


if __name__=="__main__":
    classes=["h_inv_jj_weak","z_inv_jj_qcd"]
    X_train,Y_train,X_test,Y_test=load_data(classes,input_keys=["tower_image"],suffix="low_res_tower_jet_phi",return_array=True,length=30000)












