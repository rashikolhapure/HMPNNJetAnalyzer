


from .hep.methods import DelphesNumpy
from .hep.data import NumpyEvents
from .genutils import print_events,merge_flat_dict


def get_from_indices(events,indices,keys=None):
    if keys == None: keys=list(events.keys())
    return_array={}
    for key,val in events.items():
        if key not in keys: continue
        return_array[key]=val[indices]
    return return_array





def get_delphes(run_names,**kwargs):
    if type(run_names) == str: run_names=[run_names]
    for run_name in run_names:
        now=DelphesNumpy(run_name,**kwargs)
        for events in now: 
            print_events(events)
    return 

def get_numpy_events(run_name,runs="first",**kwargs):
    now=NumpyEvents(run_name,mode="r",**kwargs)
    return_dict={}
    for item in now:
        if runs=="first": return item
        else:
            return_dict=merge_flat_dict(return_dict,item)
    return return_dict
    
