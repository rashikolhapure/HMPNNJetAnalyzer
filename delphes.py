from path import main_path
import os
import sys
sys.path.append(main_path)


import numpy as np
import matplotlib.pyplot as plt

from hep_ml.hep.methods import DelphesNumpy
from hep_ml.genutils import print_events
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-r", "--run", dest="run_name",
                  help="run_folder in madgraph_dir", metavar="directory name")
parser.add_option("-t", "--tag", dest="tag",default="",
                  help="root_file tag", metavar="directory name")
parser.add_option("-p", "--prefix", dest="delphes_prefix",default="",
                  help="run_folder in madgraph_dir", metavar="directory name")


(options,args)=parser.parse_args()
if options.run_name is None:
    print("Provide compulsory option run_name with -r <run_name> or --run <run_name> \n")
    print ("End Execution!")
    sys.exit()

d=DelphesNumpy(options.run_name,tag=options.tag,delphes_preffix=options.delphes_prefix)
for item in d: print_events(item)
