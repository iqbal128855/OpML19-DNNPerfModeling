import os
import pandas as pd
from Src.Regression import Regression
from Src.GuidedSampling import GuidedSampling

def get_data(input_file):
    """This function is used to get data
    @returns data_it, data_pc list of size number of configurations x number of configuration space
    """
    
    df=pd.read_csv(input_file)
    config_data=df[["core0_status","core1_status","core2_status","core3_status","core_freq","gpu_status","gpu_freq","emc_status","emc_freq"]]
    data_it=df[["core0_status","core1_status","core2_status","core3_status","core_freq","gpu_status","gpu_freq","emc_status","emc_freq","inference_time"]]
    data_pc=df[["core0_status","core1_status","core2_status","core3_status","core_freq","gpu_status","gpu_freq","emc_status","emc_freq","power_consumption"]]
    return (config_data,data_it,data_pc)   

if __name__=="__main__":
    output=[]  
    dir_name="/TargetData/"
    input_dir="{0}{1}".format(os.getcwd(),dir_name )
    input_data=[f for f in os.listdir(input_dir) if ((os.path.isfile(os.path.join(input_dir,f))) and (os.stat(os.path.join(input_dir,f)).st_size > 0))]
    for i in xrange (len(input_data)):
        fname=input_data[i]
        filename="{0}{1}".format(input_dir,fname)
        (conf,it,pc)=get_data(filename)
        # Inference Time
        GS_it=GuidedSampling(conf,2,"core1_status core_freq gpu_freq emc_freq core1_status:core_freq core_freq:gpu_freq core1_status:gpu_freq")
        sampled_it_index=GS_it.get_data()
        RGR_it=Regression(it,"it",sampled_it_index,)
        # Regression
        (nmse_rt_it,mape_rt_it)=RGR_it.RT()
        output.append([fname,i,"RT","IT",nmse_rt_it,mape_rt_it,"guided"])
        (nmse_rf_it,mape_rf_it)=RGR_it.RF()
        output.append([fname,i,"RF","IT",nmse_rf_it,mape_rf_it,"guided"])
        (nmse_brt_it,mape_brt_it)=RGR_it.BRT()
        output.append([fname,i,"BRT","IT",nmse_brt_it,mape_brt_it,"guided"])
        (nmse_svr_it,mape_svr_it)=RGR_it.SVR()
        output.append([fname,i,"SVR","IT",nmse_svr_it,mape_svr_it,"guided"])
        (nmse_nn_it,mape_nn_it)=RGR_it.RF()
        output.append([fname,i,"NN","IT",nmse_nn_it,mape_nn_it,"guided"])
        (nmse_mars_it,mape_mars_it)=RGR_it.MARS()
        output.append([fname,i,"MARS","IT",nmse_mars_it,mape_mars_it,"guided"])
    
        # Power Consumption
        GS_pc=GuidedSampling(conf,2,"core1_status core2_status core_freq gpu_freq emc_freq core1_status:core_freq core2_status:gpu_freq core_freq_emc_freq core_freq:gpu_freq")
        sampled_pc_index=GS_pc.get_data()
        RGR_pc=Regression(pc,"pc",sampled_pc_index)
        # Regression 
        (nmse_rt_pc,mape_rt_pc)=RGR_pc.RT()
        output.append([fname,i,"RT","PC",nmse_rt_pc,mape_rt_pc,"guided"])
        (nmse_rf_pc,mape_rf_pc)=RGR_pc.RF()
        output.append([fname,i,"RF","PC",nmse_rf_pc,mape_rf_pc,"guided"])
        (nmse_brt_pc,mape_brt_pc)=RGR_pc.BRT()
        output.append([fname,i,"BRT","PC",nmse_brt_pc,mape_brt_pc,"guided"])
        (nmse_svr_pc,mape_svr_pc)=RGR_pc.SVR()
        output.append([fname,i,"SVR","PC",nmse_svr_pc,mape_svr_pc,"guided"])
        (nmse_nn_pc,mape_nn_pc)=RGR_pc.RF()
        output.append([fname,i,"NN","PC",nmse_nn_pc,mape_nn_pc,"guided"])
        (nmse_mars_pc,mape_mars_pc)=RGR_pc.MARS()
        output.append([fname,i,"MARS","PC",nmse_mars_pc,mape_mars_pc,"guided"])
    
    df=pd.DataFrame(output)
    df.columns=["fname","env","regr","measurement","NMSE","MAPE","strategy"]
    #df.columns=["fname","env","regr","measurement","NMSE","MAPE","method"]
    if dir_name=="/TargetData/":
        df.to_csv("./Results/RQ4/GuidedSamplingTarget.csv")
    if dir_name=="/SourceData/":
        df.to_csv("./Results/RQ4/GuidedSamplingSource.csv")
    
    
    
