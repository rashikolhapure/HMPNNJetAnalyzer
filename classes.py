import os

from .hep.config import Paths
from .genutils import check_dir



class Method(object):
    def __init__(self,*args,**kwargs):
        compulsory_kwargs=("input_data","output_data")
        self.input_data=kwargs.get("input_data")
        self.output_data=kwargs.get("output_data")
        self.max_count=None
        self.count=0  

class PhysicsMethod(Method):
    def __init__(self,*args,**kwargs):
        super().__init__(args,**kwargs)
    def __iter__(self):
        return self
    def __len__(self):
        assert self.max_count,"Calling uninitialized "+type(self).__name__
        return self.max_count

class NetworkMethod(Method):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)



class Data(object):
    def __init__(self,*args,**kwargs):
        compulsory_keys={"reader_method","writer_method","run_name"}
        #assert compulsory_keys.issubset(set(kwargs.keys()))
        self.dtypes=args 
        self.prefix_path=kwargs.get("prefix_path")        
        self.reader_method=kwargs.get("reader_method")
        self.writer_method=kwargs.get("writer_method")
        self.data_ext=kwargs.get("extension")
        self.file_ext=None
        self.mg_event_path=os.path.join(Paths.madgraph_dir,kwargs["run_name"],"Events")
        self.max_count="NA"

class PhysicsData(Data):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def __iter__(self):
        return self
    def __len__(self):
        return self.max_count















