import os
import pandas as pd
import numpy as np
class ConfigurationMapping:
    def __init__(self, input_file):
        print ("Initializing ConfigurationMapping class")
        self.input_file=input_file
        self.df=pd.read_csv(self.input_file)
        self.append_new_col()
        self.assign_rank_order("core_freq","core_freq_rank")
        self.assign_rank_order("gpu_freq","gpu_freq_rank")
        self.assign_rank_order("emc_freq","emc_freq_rank")
        self.save_df()
        
        
    def append_new_col(self):
        """This function is used to append new columns
        """
        # add a new column
        length=len(self.df["core_freq"])
        self.df = self.df.assign(core_freq_rank=pd.Series(np.zeros(length)).values)
        self.df = self.df.assign(gpu_freq_rank=pd.Series(np.zeros(length)).values)
        self.df = self.df.assign(emc_freq_rank=pd.Series(np.zeros(length)).values)
    
    def assign_rank_order(self,col,new_col):
        """This function is used to assign rank to each row for core frequency, gpu frequency and memory controller frequency
        """
        # create dictiory with value as key and sorted rank as value       
        unique_val=self.df[col].unique()
        
        unique_val.sort()
        unique_dict={}
        for i in range(len(unique_val)):
            unique_dict[unique_val[i]]=i
        
        # update datafame
        for index, row in self.df.iterrows():
            self.df.at[index, new_col]=unique_dict[row[col]]
    
    def save_df(self):
        """This function is used to save dataframe
        """    
        file_name=self.input_file.split("/")
        file_name[-2]="rank"
        file_name="/".join(file_name)
        self.df.to_csv(file_name)
   
    
        
    
