from __future__ import division
import numpy as np
import math

class Metric(object):
    def __init__(self):
        print ("Initializing Metric Class")
    
    def mean_squared_error(self, Y_pred, Y_true):
        """This function is used to implement mean square error
        """
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(Y_pred,Y_true)
        
    def normalized_mean_square_error(self, Y_pred,Y_true):
        """This function is used to implement normalized mean sqaure error (NMSE)
        @returns val normalized mean square error
        """
        N=len(Y_pred)
        numerator=[Y_true[i]-Y_pred[i] for i in xrange(N)]
        numerator=[i**2 for i in numerator]
        numerator=np.sum(numerator)
        P=np.mean(Y_pred)
        M=np.mean(Y_true)
        denominator=P*M
        val=(numerator/denominator)
        val= val/N       
        return val
    
    def mean_absolute_percentage_error(self,Y_pred,Y_true):
        """This function is used to implement mean absolute percentage error
        @returns val mean absolute percentage error
        """
        N=len(Y_pred)
        numerator=[abs(Y_true[i]-Y_pred[i])/Y_true[i] for i in xrange(N)]
        numerator=np.sum(numerator)
        val=(numerator/N)*100
        return val
    
    def entropy(self,Y):
        """This function is used to implement entropy
        entropy=Summation (Pi * logPi)
        @returns entropy entropy
        """
        res=np.histogram(Y,bins="auto")[0].tolist()
        res_sum=np.sum(res)
        prob=[i/res_sum for i in res]
        prob=list(filter(lambda a: a!=0,prob))
        entropy=0
        for ind in prob:
            log_part=math.log(ind)/math.log(2)
            cur=-i*log_part
            entropy+=cur
        return entropy        
