import os
import pandas as pd
from Src.Regression import Regression
from Src.GuidedSampling import GuidedSampling

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

def perform_prediction(fname,
                       env,
                       RGR,
                       output,
                       regr,
                       measurement,
                       method): 
     if regr=="RT":
         (nmse,mape)=RGR.RT()
     if regr=="RF":
         (nmse,mape)=RGR.RF()
     if regr=="BRT":
         (nmse,mape)=RGR.BRT()
     if regr=="NN":
         (nmse,mape)=RGR.NN()
     if regr=="SVR":
         (nmse,mape)=RGR.SVR()
     if regr=="MARS":
         (nmse,mape)=RGR.MARS()
     output.append([fname,
                    env,
                    regr,
                    measurement,
                    nmse,
                    mape,
                    method])
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
    output=[]  
    dir_name="/TargetData/"
    input_dir="{0}{1}".format(os.getcwd(),dir_name )
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    for i in xrange (len(input_data)):
        fname=input_data[i]
        cur_env=fname.split("-")[-1]
        cur_env=int(cur_env.split(".")[0])
        filename="{0}{1}".format(input_dir,fname)
        (conf,it,pc)=get_data(filename)
        regressor=["RT","RF","BRT","NN","SVR","MARS"]
        # Inference Time
        GS_it=GuidedSampling(conf,2,"core1_status core2_status core_freq gpu_freq emc_freq core_freq:emc_freq core1_status:core_freq core1_status:gpu_freq core_freq:gpu_freq")
        sampled_it_index=GS_it.get_data()
        RGR_it=Regression(it,"it",sampled_it_index,)        
        # Regression
        for regr in regressor:
            output=perform_prediction(fname, 
                                      env[cur_env], 
                                      RGR_it, 
                                      output, 
                                      regr,
                                      "it",
                                      "GS")
        
        
        # Power Consumption
        GS_pc=GuidedSampling(conf,2,"core1_status core2_status core_freq gpu_freq emc_freq core_freq:emc_freq core1_status:core_freq core1_status:gpu_freq core_freq:gpu_freq")
        sampled_pc_index=GS_pc.get_data()
        RGR_pc=Regression(pc,"pc",sampled_pc_index)      
        # Regression
        for regr in regressor:
            output=perform_prediction(fname, 
                                      env[cur_env], 
                                      RGR_pc, 
                                      output, 
                                      regr,
                                      "pc",
                                      "GS")
            
    df=pd.DataFrame(output)
    df.columns=["fname","Environment","Regression","Measurement","NMSE","MAPE","Method"]
    if dir_name=="/TargetData/":
        df.to_csv("./Results/RQ4/GuidedSamplingTarget.csv")
    if dir_name=="/SourceData/":
        df.to_csv("./Results/RQ4/GuidedSamplingTarget.csv")
    
    
     
