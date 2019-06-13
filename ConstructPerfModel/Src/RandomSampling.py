import random
import numpy as np
from operator import itemgetter

class RandomSampling(object):
    """This class is used to implement different types of random sampling techniques
    """
    def __init__(self,data,num):
        print ("Initialzing Random Sampling Class")
        self.data=data
        self.num=num
        
    def simple_random_sampling(self):
        """This function is used to implement simple random sampling
        """
        return random.sample(self.data,self.num)
    
    def uniform_random_sampling(self):
        """This function is used to implement uniform random sampling
        """
        random_data=[]
        cur=np.random.uniform(0,len(self.data),self.num)
        cur=[int(i) for i in cur]
        for i in xrange(len(cur)):
            random_data.append(self.data[i])
        return random_data
            
    def systematic_random_sampling(self):
        """This function is used to implement systematic random sampling 
        """
        N=len(self.data)
        k=N/self.num
        index=0
        iteration=0
        random_data=[]
        while (iteration< self.num):
            random_data.extend(random.sample(self.data[index:index+k],1))
            iteration+=1
            index+=k
        
        return random_data
        
    def stratified_random_sampling(self):
        """This function is used to implement stratified random sampling
        """
        sorted_data=sorted(self.data,key=itemgetter(9))
        random_data=random.sample(sorted_data[0:self.num/2],self.num/2)
        random_data.extend(random.sample(sorted_data[(self.num/2)+1:self.num],(self.num/2)-1))
        return random_data
     
    def multistage_random_sampling(self):
        """This function is used to implement multistage random sampling
        """
        num_cluster=8
        random_data=[]
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=(num_cluster), precompute_distances=True, ).fit(self.data)
        selected_cluster=random.randint(0,num_cluster-1)
        labels=kmeans.labels_.tolist()
        for i in xrange(len(labels)):
            if labels[i]==selected_cluster:
                random_data.append(self.data[i])
        return random_data
    
    
