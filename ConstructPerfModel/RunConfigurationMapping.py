import os
import pandas as pd
from Src.ConfigurationMapping import ConfigurationMapping

if __name__=="__main__":
     
    # Get Input
    dir_name="/TX2/data/"
    input_dir="{0}{1}".format(os.getcwd(),dir_name )
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    # Perform Configuration Mapping
    for data in input_data:
       current_data=os.path.join(input_dir,data)       
       ConfigurationMapping(current_data)
       
       
        
