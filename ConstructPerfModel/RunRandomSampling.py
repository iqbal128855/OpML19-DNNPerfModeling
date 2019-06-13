import os
import pandas as pd
from Src.Regression import Regression
from Src.RandomSampling import RandomSampling

def get_data(input_file):
    """This function is used to get data
    @returns data_it, data_pc list of size number of configurations x number of configuration space
    """
    
    df=pd.read_csv(input_file)
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
    return (data_it,data_pc)     

def perform_random_sampling(RS_obj,
                            fname,
                            environment,
                            data,
                            mode,
                            sampling_type):
    """This function is used to measure performance for random sampling
    """
    output=list()
    regressor=["RT","RF","BRT","SVR","NN","MARS"]
    if sampling_type=="Simple": random_data=RS_obj.simple_random_sampling()
    if sampling_type=="Uniform": random_data=RS_obj.uniform_random_sampling()
    if sampling_type=="Stratified": random_data=RS_obj.stratified_random_sampling()
    if sampling_type=="Systematic": random_data=RS_obj.systematic_random_sampling()
    if sampling_type=="Multistage": random_data=RS_obj.multistage_random_sampling()
    random_data=pd.DataFrame(random_data)  
    sampled_X=random_data.iloc[:, 0:9].values.tolist()
    sampled_Y=random_data.iloc[:, -1].values.tolist()
    
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
    # Initialize
    simple_random_output=list()
    uniform_random_output=list()
    stratified_random_output=list()
    systematic_random_output=list()
    multistage_random_output=list()
    # Get Input
    dir_name="/TargetData/"
    input_dir="{0}{1}".format(os.getcwd(),dir_name )
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    
    for iteration in xrange (len(input_data)):
        fname=input_data[iteration]
        filename="{0}{1}".format(input_dir,fname)
        (it,pc)=get_data(filename)
        cur_env=fname.split("-")[-1]
        cur_env=int(cur_env.split(".")[0]) 
        if cur_env<8:
            dlength=45
        else:
            dlength=84
         
        # Inference Time        
        it_list=it.values.tolist()
        RS_it=RandomSampling(it_list,dlength)
        # Power Consumption      
        pc_list=pc.values.tolist()
        RS_pc=RandomSampling(pc_list,dlength)  
               
        for i in xrange(10):
            # Simple random
            
            simple_random_output.extend(perform_random_sampling(RS_it,
                                                                fname,
                                                                env[cur_env],
                                                                it,
                                                                "it",
                                                                "Simple"))  
            simple_random_output.extend(perform_random_sampling(RS_pc,
                                                               fname,
                                                               env[cur_env],
                                                               pc,
                                                               "pc",
                                                               "Simple"))
            # Uniform Random    
            uniform_random_output.extend(perform_random_sampling(RS_it,
                                                                fname,
                                                                env[cur_env],
                                                                it,
                                                                "it",
                                                                "Uniform"))
            uniform_random_output.extend(perform_random_sampling(RS_pc,
                                                                fname,
                                                                env[cur_env],
                                                                pc,
                                                                "pc",
                                                                "Uniform"))   
            # Stratified Random                                                       
            stratified_random_output.extend(perform_random_sampling(RS_it,
                                                                    fname,
                                                                    env[cur_env],
                                                                    it,
                                                                    "it",
                                                                    "Stratified"))
            stratified_random_output.extend(perform_random_sampling(RS_pc,
                                                                    fname,
                                                                    env[cur_env],
                                                                    pc,
                                                                    "pc",
                                                                    "Stratified"))
            
            # Systematic Random                                                        
            systematic_random_output.extend(perform_random_sampling(RS_it,
                                                                    fname,
                                                                    env[cur_env],
                                                                    it,
                                                                    "it",
                                                                    "Systematic"))
            systematic_random_output.extend(perform_random_sampling(RS_pc,
                                                                    fname,
                                                                    env[cur_env],
                                                                    pc,
                                                                    "pc",
                                                                    "Systematic"))
            
            # MultiStage Random                                                                 
            multistage_random_output.extend(perform_random_sampling(RS_it,
                                                                    fname,
                                                                    env[cur_env],
                                                                    it,
                                                                    "it",
                                                                    "Multistage"))
            multistage_random_output.extend(perform_random_sampling(RS_pc,
                                                                    fname,
                                                                    env[cur_env],
                                                                    pc,
                                                                    "pc",
                                                                    "Multistage"))
           
                  
    save_file(dir_name,simple_random_output,"SimpleRandomSampling")
    save_file(dir_name,uniform_random_output,"UniformRandomSampling")
    save_file(dir_name,stratified_random_output,"StratifiedRandomSampling")
    save_file(dir_name,systematic_random_output,"SystematicRandomSampling")
    save_file(dir_name,multistage_random_output,"MultiStageRandomSampling")
    
   
    
    
        
