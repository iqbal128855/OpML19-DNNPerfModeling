import os
import pickle
import pandas as pd
from Src.Metric import Metric
from Src.RegressionSource import RegressionSource

def get_data(input_file):
    """This function is used to get data
    @returns data_it, data_pc list of size number of configurations x number of configuration space
    """
    
    df=pd.read_csv(input_file)
    config_data=df[["core0_status","core1_status","core2_status","core3_status","core_freq","gpu_status","gpu_freq","emc_status","emc_freq"]]
    data_it=df[["core0_status","core1_status","core2_status","core3_status","core_freq","gpu_status","gpu_freq","emc_status","emc_freq","inference_time"]]
    data_pc=df[["core0_status","core1_status","core2_status","core3_status","core_freq","gpu_status","gpu_freq","emc_status","emc_freq","power_consumption"]]
    return (config_data,data_it,data_pc)     


def get_X_Y(data):
    return (data.iloc[:, 0:9].values.tolist(),
            data.iloc[:, -1].values.tolist())

def perform_prediction(fname,
                       env,
                       output,
                       regr,
                       measurement,
                       method):
    """This function is used to learn model, compute metric and return output
    """
    model=pickle.load(open("./Model/ModelTransfer/"+regr+"_"+measurement+".sav","rb"))
    Y_pred=model.predict(X)
    nmse=metric.normalized_mean_square_error(Y_pred,Y)
    mape=metric.mean_absolute_percentage_error(Y_pred,Y) 
    output.append([fname,env,regr,measurement,nmse,mape,method])
    return output 

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
    dir_name="/SourceData/"
    input_dir="{0}{1}".format(os.getcwd(),dir_name )
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    # Get Input 
    for fname in input_data:
        filename="{0}{1}".format(input_dir,fname)
        (conf,it,pc)=get_data(filename)
    
    RegressionSource(it,"it")
    RegressionSource(pc,"pc")
    # Get Input Files   
    dir_name="/TargetData/"
    input_dir="{0}{1}".format(os.getcwd(),dir_name )
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    # Initialize
    metric=Metric()
    output=[]
    regressor=["RT","NN","RF","BRT","SVR","MARS"]
    
    for i in xrange (len(input_data)):
        fname=input_data[i]  
        filename="{0}{1}".format(input_dir,fname)
        (conf,it,pc)=get_data(filename)
        cur_env=fname.split("-")[-1]
        cur_env=int(cur_env.split(".")[0])
        
        # Inference Time
        (X,Y)=get_X_Y(it)
        for regr in regressor:
            output=perform_prediction(fname,
                                    env[cur_env],
                                    output,
                                    regr,
                                    "it",
                                    "DM")
        # Power Consumption
        (X,Y)=get_X_Y(pc)
        
        for regr in regressor:
            output=perform_prediction(fname,
                                    env[cur_env],
                                    output,
                                    regr,
                                    "pc",
                                    "DM")
          
    df=pd.DataFrame(output)
    df.columns=["fname","Environment","Regression","Measurement","NMSE","MAPE","Method"]
    if dir_name=="/TargetData/":
        df.to_csv("./Results/RQ5/ModelTransferTarget.csv")
    if dir_name=="/SourceData/":
        df.to_csv("./Results/RQ5/ModelTransferSource.csv")     
        
        
        
    
        
