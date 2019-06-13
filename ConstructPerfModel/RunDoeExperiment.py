import os
import csv
from pyDOE import *
from Src.pyDOE_corrected import *
from diversipy import *
from Src.Regression import Regression
import pandas as pd
import numpy as np

def construct_df(x,r):
    """This function is used to construct dataframe
    """
    df=pd.DataFrame(data=x,dtype='float32')
    for i in df.index:
        for j in range(len(list(df.iloc[i]))):
            df.iloc[i][j]=r[j][int(df.iloc[i][j])]
    return df

def construct_df_from_matrix(x,factor_array):
    """This function is used to construct dataframe from matrix
    """   
    row_num=x.shape[0] 
    col_num=x.shape[1] 
    empty=np.zeros((row_num,col_num))  
    
    def simple_substitution(idx,factor_list):
        if idx==-1:
            return factor_list[0]
        elif idx==0:
            return factor_list[1]
        elif idx==1:
            return factor_list[2]
        else:
            alpha=np.abs(factor_list[2]-factor_list[0])/2
            if idx<0:
                beta=np.abs(idx)-1
                return factor_list[0]-(beta*alpha)
            else:
                beta=idx-1
                return factor_list[2]+(beta*alpha)
        
    for i in range(row_num):
        for j in range(col_num):
            empty[i,j] = simple_substitution(x[i,j],factor_array[j])
        
    return pd.DataFrame(data=empty)

def construct_df_from_random_matrix(x,factor_array):
    """This function is used to construct dataframe from random matrix
    """    
    row_num=x.shape[0] 
    col_num=x.shape[1] 
    
    empty=np.zeros((row_num,col_num))  
    
    def simple_substitution(idx,factor_list):
        alpha=np.abs(factor_list[1]-factor_list[0])
        beta=idx
        return factor_list[0]+(beta*alpha)
        
    for i in range(row_num):
        for j in range(col_num):
            empty[i,j] = simple_substitution(x[i,j],factor_array[j])
        
    return pd.DataFrame(data=empty)

def build_box_behnken(factor_level_ranges,center=1):
    """This function is used to implement box behnken design
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])==2:
            factor_level_ranges[key].append((factor_level_ranges[key][0]+factor_level_ranges[key][1])/2)
            factor_level_ranges[key].sort()
                
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = bbdesign_corrected(factor_count,center=center)
    x=x+1 

    df=construct_df(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    
    return df

def build_random_k_means(factor_level_ranges, num_samples=None):
    """This function is used to implement random k means design
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
             
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    if num_samples==None:
        num_samples=factor_count
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = random_k_means(num_points=num_samples,dimension=factor_count) 
    factor_lists=np.array(factor_lists)
    
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

def build_central_composite(factor_level_ranges,center=(2,2),alpha='o',face='ccc'):
    """This function is used to implement central composite design
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
                
    # Creates the mid-points by averaging the low and high levels
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])==2:
            factor_level_ranges[key].append((factor_level_ranges[key][0]+factor_level_ranges[key][1])/2)
            factor_level_ranges[key].sort()
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = ccdesign(factor_count,center=center,alpha=alpha,face=face)
    factor_lists=np.array(factor_lists)
    
    df = construct_df_from_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df

def build_lhs(factor_level_ranges, num_samples=None, prob_distribution=None):
    """This function is used to implement latin hypercube design
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
                
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    if num_samples==None:
        num_samples=factor_count
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = lhs(n=factor_count,samples=num_samples)
    factor_lists=np.array(factor_lists)
    
    df = construct_df_from_random_matrix(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    return df  

def build_plackett_burman(factor_level_ranges):
    """This function is used to implement placker burman design
    """
    for key in factor_level_ranges:
        if len(factor_level_ranges[key])!=2:
            factor_level_ranges[key][1]=factor_level_ranges[key][-1]
            factor_level_ranges[key]=factor_level_ranges[key][:2]
    
    factor_count=len(factor_level_ranges)
    factor_lists=[]
    
    for key in factor_level_ranges:
        factor_lists.append(factor_level_ranges[key])
    
    x = pbdesign(factor_count)
    
    def index_change(x):
        if x==-1:
            return 0
        else:
            return x
    vfunc=np.vectorize(index_change)
    x=vfunc(x)
       
    df=construct_df(x,factor_lists)
    df.columns=factor_level_ranges.keys()
    
    return df

def read_variables_csv(csvfile):
    """This function is used to preprocess data
    """
    dict_key={}
    try:
        with open(csvfile) as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            for field in fields:
                lst=[]
                with open(csvfile) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lst.append(float(row[field]))
                dict_key[field]=lst
    
        return dict_key
    except:
        print("Error in reading the specified file from the disk. Please make sure it is in current directory.")
        return -1

def get_data(input_file):
    """This function is used to get data
    @returns data_it, data_pc list of size number of configurations x number of configuration space
    """
    
    df=pd.read_csv(input_file)
    config_data=df[["core0_status",
                    "core1_status",
                    "core2_status",
                    "core3_status",
                    "core_freq",
                    "gpu_status",
                    "gpu_freq",
                    "emc_status",
                    "emc_freq"]]
    data_it=df[["core0_status",
                "core1_status",
                "core2_status",
                "core3_status",
                "core_freq",
                "gpu_status",
                "gpu_freq",
                "emc_status",
                "emc_freq",
                "inference_time"]]
    data_pc=df[["core0_status",
                "core1_status",
                "core2_status",
                "core3_status",
                "core_freq",
                "gpu_status",
                "gpu_freq",
                "emc_status",
                "emc_freq",
                "power_consumption"]]
    return (config_data,data_it,data_pc)     

def compute_sampling_performance(dict_vars,
                                 fname,
                                 environment,
                                 data,
                                 mode,
                                 sampling_type):
    """This function is used to compute performance
    """
    
    output=[]
    regressor=["RT","RF","BRT","NN","SVR","MARS"]
    if sampling_type=="PB":
        df=build_plackett_burman(dict_vars)
    if sampling_type=="CC":
        df=build_central_composite(dict_vars,face='ccc')
    if sampling_type=="BB":
        df=build_box_behnken(dict_vars)
    if sampling_type=="RKM":
        df=build_random_k_means(dict_vars)
    if sampling_type=="LHS":
        df=build_lhs(dict_vars)
    if mode=="it":
        df = df[["core0_status",
                "core1_status",
                "core2_status",
                "core3_status",
                "core_freq",
                "gpu_status",
                "gpu_freq",
                "emc_status",
                "emc_freq",
                "inference_time"]]
    if mode=="pc":
        df = df[["core0_status",
                "core1_status",
                "core2_status",
                "core3_status",
                "core_freq",
                "gpu_status",
                "gpu_freq",
                "emc_status",
                "emc_freq",
                "power_consumption"]]
    sampled_X=df.iloc[:, 0:9].values.tolist()
    sampled_Y=df.iloc[:, -1].values.tolist()
    print (len(sampled_X))
    RGR=Regression(data,mode)
    for regr in regressor:
        if regr=="RT": (nmse,mape)= RGR.RT(sampled_X,sampled_Y)
        if regr=="BRT": (nmse,mape)=RGR.BRT(sampled_X,sampled_Y)       
        if regr=="RF": (nmse,mape)=RGR.RF(sampled_X,sampled_Y)       
        if regr=="NN": (nmse,mape)=RGR.NN(sampled_X,sampled_Y)       
        if regr=="SVR": (nmse,mape)=RGR.SVR(sampled_X,sampled_Y)       
        if regr=="MARS": (nmse,mape)=RGR.MARS(sampled_X,sampled_Y)               
        
        output.append([fname,
                       environment,
                       regr,
                       mode,
                       nmse,
                       mape,
                       sampling_type])
    return output        
   

def save_file(dir_name,
               output,
               technique):
     """This function is used to save file
     """   
     cols=["fname","Environment","Regression","Measurement","NMSE","MAPE","Method"]
     df=pd.DataFrame(output)
     df.columns=cols
     if dir_name=="/TargetData/":
         df.to_csv("./Results/RQ4/"+str(technique)+"Target.csv")
     if dir_name=="/SourceData/":
         df.to_csv("./Results/RQ4/"+str(technique)+"Source.csv")

if __name__=="__main__":
     # env dictionary
    env={
          0:"(h1,m1,s1)",
          1:"(h1,m1,s2)",
          2:"(h1,m1,s3)",
          3:"(h1,m1,s4)",
          4:"(h1,m2,s1)",
          5:"(h1,m2,s2)",
          6:"(h1,m2,s3)",
          7:"(h1,m2,s4)",
          8:"(h2,m1,s1)",
          9:"(h2,m1,s2)",
         10:"(h2,m1,s3)",
         11:"(h2,m1,s4)",
         12:"(h2,m2,s1)",
         13:"(h2,m2,s2)",
         14:"(h2,m2,s3)",
         15:"(h2,m2,s4)"
         }
    # Get Input
    dir_name="/TargetData/"
    input_dir="{0}{1}".format(os.getcwd(),dir_name )
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    pb_output=list()
    cc_output=list()
    bb_output=list()
    rkm_output=list()
    lhs_output=list()
    for i in xrange (len(input_data)):
        fname=input_data[i]
        filename="{0}{1}".format(input_dir,fname)
        (conf,it,pc)=get_data(filename)

        cur_env=fname.split("-")[-1]
        cur_env=int(cur_env.split(".")[0]) 
        it.to_csv("./Temp/processed"+"_"+fname+"it.csv")
        fname_processed_it="./Temp/processed"+"_"+fname+"it.csv"
        
        pc.to_csv("./Temp/processed"+"_"+fname+"pc.csv")
        fname_processed_pc="./Temp/processed"+"_"+fname+"pc.csv"
        
        # Inference Time
        dict_vars_it=read_variables_csv(fname_processed_it)
        if type(dict_vars_it)!=int:
            factor_count=len(dict_vars_it)
        
        # PB
        pb_output.extend(compute_sampling_performance(dict_vars_it,
                                                      fname,
                                                      env[cur_env],
                                                      it,
                                                      "it",
                                                      "PB"))
        # BB
        bb_output.extend(compute_sampling_performance(dict_vars_it,
                                                      fname,
                                                      env[cur_env],
                                                      it,
                                                      "it",
                                                      "BB"))
        
        # CC
        cc_output.extend(compute_sampling_performance(dict_vars_it,
                                                      fname,
                                                      env[cur_env],
                                                      it,
                                                      "it",
                                                      "CC"))
        # rkm
        for i in xrange(10):
            rkm_output.extend(compute_sampling_performance(dict_vars_it,
                                                           fname,
                                                           env[cur_env],
                                                           it,
                                                           "it",
                                                           "RKM"))
        for i in xrange(10):
            lhs_output.extend(compute_sampling_performance(dict_vars_it,
                                                           fname,
                                                           env[cur_env],
                                                           it,
                                                           "it",
                                                           "LHS"))
        
        # Power Consumption
        dict_vars_pc=read_variables_csv(fname_processed_pc)
        if type(dict_vars_pc)!=int:
            factor_count=len(dict_vars_pc)
            
        pb_output.extend(compute_sampling_performance(dict_vars_pc,
                                                     fname,
                                                     env[cur_env],
                                                     pc,
                                                     "pc",
                                                     "PB"))
        
        # BB        
        bb_output.extend(compute_sampling_performance(dict_vars_pc,
                                                      fname,
                                                      env[cur_env],
                                                      pc,
                                                      "pc",
                                                      "BB"))
        
        # CC
        cc_output.extend(compute_sampling_performance(dict_vars_pc,
                                                      fname,
                                                      env[cur_env],
                                                      pc,
                                                      "pc",
                                                      "CC"))
        
        # RKM
        for i in xrange(10):
            rkm_output.extend(compute_sampling_performance(dict_vars_pc,
                                                           fname,
                                                           env[cur_env],
                                                           pc,
                                                           "pc",
                                                           "RKM"))
        
        # LHS
        for i in xrange(10):
            lhs_output.extend(compute_sampling_performance(dict_vars_pc,
                                                           fname,
                                                           env[cur_env],
                                                           pc,
                                                           "pc",
                                                           "LHS"))
        
        
    save_file(dir_name,pb_output,"PlacketBurman")
    
    save_file(dir_name,cc_output,"CentralComposite")
    save_file(dir_name,bb_output,"BoxBehnken")
    save_file(dir_name,rkm_output,"RandomKMeans")
    save_file(dir_name,lhs_output,"LatinHypercubeSampling")
    
    
    
    
    
