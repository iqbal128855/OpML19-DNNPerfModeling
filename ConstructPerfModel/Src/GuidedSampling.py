import itertools
import pandas as pd

class GuidedSampling(object):
    """This class is used to implement the guided sampling technique
    """
    def __init__(self,data,order,equation):
        print ("Initialzing Guided Sampling Class")
        self.order=2
        self.equation=equation
        self.data=data
        self.DEFAULT=0
        self.params=[]
        self.col=["core0_status","core1_status","core2_status","core3_status","core_freq","gpu_status","gpu_freq","emc_status","emc_freq"]
        self.default_core0_status=1
        self.default_core1_status=1
        self.default_core2_status=1
        self.default_core3_status=1
        self.default_core_freq=self.data.iloc[:,4].values[self.DEFAULT]
        self.default_gpu_status=self.data.iloc[:,5].values[self.DEFAULT]
        self.default_gpu_freq=self.data.iloc[:,6].values[self.DEFAULT]
        self.default_emc_status=self.data.iloc[:,7].values[self.DEFAULT]
        self.default_emc_freq=self.data.iloc[:,8].values[self.DEFAULT]
        
        # initial hash table 
        self.header={
                     "core0_status":{"update":False,
                                     "value":[self.default_core0_status]},
                     "core1_status":{"update":False,
                                     "value":[self.default_core1_status]},
                     "core2_status":{"update":False,
                                     "value":[self.default_core2_status]},
                     "core3_status":{"update":False,
                                     "value":[self.default_core3_status]},
                     "core_freq":{"update":False,
                                  "value":[self.default_core_freq]},
                     "gpu_status":{"update":False,
                                   "value":[self.default_gpu_status]},
                     "gpu_freq":{"update":False,
                                 "value":[self.default_gpu_freq]},
                     "emc_status":{"update":False,
                                   "value":[self.default_emc_status]},
                     "emc_freq":{"update":False,
                                 "value":[self.default_emc_freq]}
                     }
        
        self.interactions=self.equation.split(" ")
        self.var=[_ for _ in xrange(len(self.col))]
        for iaction in self.interactions:
            if ":" in iaction:
                cur=iaction.split(":")
                # TODO
                self.var[self.col.index(cur[0])]=list(set(self.data[cur[0]].values))
                self.var[self.col.index(cur[1])]=list(set(self.data[cur[1]].values))
                self.header[cur[0]]["update"]=True
                self.header[cur[1]]["update"]=True
                #print ("Interaction Term: ",iaction)
                self.set_value()                   
                self.reinitialize()
            else:
                #print ("Interaction Term: ",iaction)
                if ((iaction=="core_freq") or 
                    (iaction=="gpu_freq") or 
                    (iaction=="emc_freq")):
                    self.var[self.col.index(iaction)]=list(set(self.data[iaction].values))
                    self.header[iaction]["update"]=True
                self.set_value()        
                self.reinitialize()
        
    
    def set_value(self):
        """This function is used to set values
        """
        
        for option in self.col:
            if self.header[option]["update"] is False:
                self.var[self.col.index(option)]=self.header[option]["value"]
            
        cur_params=list(itertools.product(*self.var))
        #print ("Current Params: ",cur_params)
        for i in cur_params:
            self.params.append(list(i))
                
    def reinitialize(self):
        """This function is reset self.header 
        """
        for i in self.header:
            self.header[i]["update"]=False
            
    def get_data(self):
        """This function is used to get index
        """
        
        self.ind=[]
        data=self.data.values.tolist()
        for i in self.params:
            try:
                self.ind.append(data.index(i))
            except ValueError:
                continue
        self.ind=list(set(self.ind)) 
        print (len(self.ind))   
       
        return self.ind
        
             


