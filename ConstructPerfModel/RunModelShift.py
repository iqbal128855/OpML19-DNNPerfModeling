import os
import pickle
import pandas as pd
import random
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from Src.Metric import Metric
from Src.RegressionSource import RegressionSource

def get_data(input_file):
    """This function is used to get data
    @returns data_it, data_pc list of size number of configurations x number of configuration space
    """ 
    df=pd.read_csv(input_file)   
    # Inference Time Data
    data_it=df[["core0_status",
                "core1_status",
                "core2_status",
                "core3_status",
                "core_freq",
                "gpu_status",
                "gpu_freq",
                "emc_status",
                "emc_freq",
                "core_freq_rank",
                "gpu_freq_rank",
                "emc_freq_rank",
                "inference_time"]]
    # Power Consuption Data
    data_pc=df[["core0_status",
                "core1_status",
                "core2_status",
                "core3_status",
                "core_freq",
                "gpu_status",
                "gpu_freq",
                "emc_status",
                "emc_freq",
                "core_freq_rank",
                "gpu_freq_rank",
                "emc_freq_rank",
                "power_consumption"]]
    return (data_it,data_pc)     

def learn_linear_transfer(source_Y,target_Y):
        """This function is used to perform a transfer function
        """      
        regr = linear_model.LinearRegression()
        s=[[i] for i in source_Y]
        t=[[i] for i in target_Y]
        regr.fit(s,t)
        return regr

def pre_process(x):
    """This function is used to preprocess data
    """
    return [[i] for i in x]

def learn_non_linear_transfer(source_Y,target_Y):
        """This function is used to perform a transfer function
        """      
        regr = RandomForestRegressor(n_estimators=16)
        
        s=[[i] for i in source_Y]
        t=[i for i in target_Y]
        regr.fit(s,t)        
        return regr
        
def get_X_Y(data):
    return (data.iloc[:, 0:9].values.tolist(),
            data.iloc[:, -1].values.tolist())

def perform_prediction(fname,
                       env,
                       output,
                       regr,
                       measurement,
                       method,
                       source_data):
    """This function is used to learn model, compute metric and return output
    """
    model=pickle.load(open("./Model/ModelTransfer/"+regr+"_"+measurement+".sav","rb"))
    Y_pred_model=model.predict(X)
    source_Y=source_data    
    if method=="LMS":
        ls=learn_linear_transfer(source_Y,target_Y)
    if method=="NLMS":
        ls=learn_non_linear_transfer(source_Y,target_Y)
    Y_pred=ls.predict(pre_process(Y_pred_model))
    nmse=metric.normalized_mean_square_error(Y_pred,Y)
    mape=metric.mean_absolute_percentage_error(Y_pred,Y) 
    output.append([fname,env,regr,measurement,nmse,mape,method])
    return output 

def delete_by_indices(lst, indices):
    indices_as_set = set(indices)
    return [ lst[i] for i in xrange(len(lst)) if i not in indices_as_set ]

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
                     
    # Train 
    dir_name="/SourceData/"
    input_dir="{0}{1}".format(os.getcwd(),dir_name )
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    # Get Input
    output=[]
    for iteration in xrange(10): 
        for fname in input_data:
            filename="{0}{1}".format(input_dir,fname)
            (source_it,source_pc)=get_data(filename)
        (source_X_it,source_Y_it)=get_X_Y(source_it)
        (source_X_pc,source_Y_pc)=get_X_Y(source_pc)
     
        # Test 
        dir_name="/TargetData/"
        input_dir="{0}{1}".format(os.getcwd(),dir_name )
        input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))] 
        metric=Metric()
        regressor=["RT","NN","RF","BRT","SVR","MARS"] 
        for i in range(len(input_data)):
            fname=input_data[i]    
            filename="{0}{1}".format(input_dir,fname)
            (it,pc)=get_data(filename)
            cur_env=fname.split("-")[-1]
            cur_env=int(cur_env.split(".")[0]) 
            # Inference Time       
            if cur_env<8:
                cur_it=it.drop(["core_freq","gpu_freq","emc_freq","inference_time"],axis=1).values.tolist()
                s_it=source_it.drop(["core_freq","gpu_freq","emc_freq","inference_time"],axis=1).values.tolist()
                index, del_index=[],[]
                for conf in xrange(len(s_it)):
                    try:
                        index.append(cur_it.index(s_it[conf]))
                    except ValueError:
                        del_index.append(conf)
                        continue  
                source_X_data=delete_by_indices(source_X_it,del_index)
                source_Y_data=delete_by_indices(source_Y_it,del_index)   
                                         
            else:
                cur_it=it.drop("inference_time",axis=1).values.tolist()
                s_it=source_it.drop("inference_time",axis=1).values.tolist()         
                index=[cur_it.index(conf) for conf in s_it]                  
                source_X_data=source_X_it
                source_Y_data=source_Y_it
            
            (X,Y)=get_X_Y(it)
            target_X=[X[conf] for conf in index]
            target_Y=[Y[conf] for conf in index]
            for regr in regressor:
                output=perform_prediction(fname,
                                      env[cur_env],
                                      output,
                                      regr,
                                      "it",
                                      "LMS",
                                      source_Y_data)
                output=perform_prediction(fname,
                                      env[cur_env],
                                      output,
                                      regr,
                                      "it",
                                      "NLMS",
                                      source_Y_data)
        
            # Power Consumption         
            if cur_env<8:
                cur_pc=pc.drop(["core_freq","gpu_freq","emc_freq","power_consumption"],axis=1).values.tolist()
                s_pc=source_pc.drop(["core_freq","gpu_freq","emc_freq","power_consumption"],axis=1).values.tolist()
                index, del_index=[],[]
                for conf in xrange(len(s_pc)):
                    try:
                        index.append(cur_pc.index(s_pc[conf]))
                    except ValueError:
                        del_index.append(conf)
                        continue  
                source_X_data=delete_by_indices(source_X_pc,del_index)
                source_Y_data=delete_by_indices(source_Y_pc,del_index)   
                
            else:
                cur_pc=pc.drop("power_consumption",axis=1).values.tolist()
                s_pc=source_pc.drop("power_consumption",axis=1).values.tolist()
                index=[cur_pc.index(conf) for conf in s_pc]
                source_X_data=source_X_pc
                source_Y_data=source_Y_pc
        
            (X,Y)=get_X_Y(pc)
            target_X=[X[conf] for conf in index]
            target_Y=[Y[conf] for conf in index]
            for regr in regressor:
                output=perform_prediction(fname,
                                      env[cur_env],
                                      output,
                                      regr,
                                      "pc",
                                      "LMS",
                                      source_Y_data)
                output=perform_prediction(fname,
                                      env[cur_env],
                                      output,
                                      regr,
                                      "pc",
                                      "NLMS",
                                      source_Y_data)
        
    df=pd.DataFrame(output)
    df.drop_duplicates(keep=False, inplace=True)
    df.columns=["fname","Environment","Regression","Measurement","NMSE","MAPE","Method"]
    if dir_name=="/TargetData/":
        df.to_csv("./Results/RQ5/ModelShiftTarget.csv")
    if dir_name=="/SourceData/":
        df.to_csv("./Results/RQ5/ModelShiftSource.csv")     
    
        
        
    
        
