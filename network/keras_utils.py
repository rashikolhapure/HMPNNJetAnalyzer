
import tensorflow.keras as keras
import numpy as np

from itertools import product
import sys


def opt(learning_rate,optimizer_name,**kwargs):
    opt_dict={"Adam":0,"Adamax":1,"Nadam":2,"Adadelta":3,"Adagrad":4,"RMSprop":5,"SGD":6,"Adadelta":7}
    print (optimizer_name," initialized with learning rate ",learning_rate)
    if opt_dict[optimizer_name]==0:
        return keras.optimizers.Adam(lr=learning_rate, **kwargs)
    elif opt_dict[optimizer_name]==1:
        return keras.optimizers.Adamax(lr=learning_rate, **kwargs)
    elif opt_dict[optimizer_name]==2:
        return keras.optimizers.Nadam(lr=learning_rate, **kwargs)
    elif opt_dict[optimizer_name]==3:
        return keras.optimizers.Adadelta(lr=learning_rate, **kwargs)
    elif opt_dict[optimizer_name]==4:
        return keras.optimizers.Adagrad(lr=learning_rate, **kwargs)
    elif opt_dict[optimizer_name]==5:
        return keras.optimizers.RMSprop(lr=learning_rate, **kwargs)
    elif opt_dict[optimizer_name]==6:    
        return keras.optimizers.SGD(lr=learning_rate, **kwargs)
    elif opt_dict[optimizer_name]==7:
        return keras.optimizers.Adadelta(lr=learning_rate, **kwargs)
    else:
        raise ValueError("Wrong optimizer choice ...")


def array_shuffle(*args):      # X.shape[0]==Y.shape[0]
    ind_map=np.arange(args[0].shape[0])
    #np.random.seed(seed=random_seed)
    np.random.shuffle(ind_map)
    args=list(args)
    for i in range(len(args)):
        args[i]=args[i][ind_map]
        #print (args[i][:10])
    return tuple(args),ind_map

def get_hyper_opt_kwargs(**kwargs):
    keys=[]
    values=[]
    for key,val in kwargs.items():
        keys.append(key),values.append(val)
    prod=product(*[kwargs.get(key) for key in kwargs])
    return_list=[]
    for item in prod: 
        return_list.append({key:val for key,val in zip(keys,item)})
    return return_list
    
    
             




     
