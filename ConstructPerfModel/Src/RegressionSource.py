import pickle
from Metric import Metric

class RegressionSource(object):
    """This class is used to implement 6 different regression techniques
       
       1. Regression Tree (RT)
       2. Random Forest (RF)
       3. Boosted Regression Tree (BRT)
       4. Support Vector Regressor (SVR)
       5. Neural Network (NN)
       6. Multivariate Adaptive Regression Splines (MARS)
    """
    
    def __init__(self,
                 data,
                 mode,
                 sampled_data_index=None):
        
        self.data=data
        self.mode=mode      
        self.metric=Metric()
        
        # Original Data
        self.X=self.data.iloc[:, 0:9].values.tolist()
        self.Y=self.data.iloc[:, -1].values.tolist()
        
        
        (self.sampled_X, self.sampled_Y)=([_ for _ in xrange(len(self.X))],[_ for _ in xrange(len(self.Y))])
        
        for i in xrange(len(self.sampled_X)):    
            self.sampled_X[i]=self.X[i]
            self.sampled_Y[i]=self.Y[i]
        
        
        self.RF()
        self.NN()
        self.RT()
        self.BRT()
        self.SVR()
        self.MARS()
        
    
    def get_sampled_data(self):
        """This function is used to return sampled data
        @returns sampled_X, ssampled_Y
        """
        return (self.sampled_X, 
                self.sampled_Y)
                           
    def RT(self,
           X=None,
           Y=None):
        """This function is used to implement a regression tree
        """
        from sklearn.tree import DecisionTreeRegressor
        rgr=DecisionTreeRegressor(random_state=20)
        
        if (X is not None and 
            Y is not None):
            (self.sampled_X,self.sampled_Y)=(X,Y)
        # train    
        rgr.fit(self.sampled_X,self.sampled_Y)
        filename = './Model/ModelTransfer/RT_'+self.mode+'.sav'
        pickle.dump(rgr, open(filename, 'wb'))       
        
    
    def RF(self,
           X=None,
           Y=None):
        """This function is used to implement a random forest
        """
        from sklearn.ensemble import RandomForestRegressor
        rgr=RandomForestRegressor(n_estimators=16)
        if (X is not None and 
            Y is not None):
            (self.sampled_X,self.sampled_Y)=(X,Y)
        # train    
        rgr.fit(self.sampled_X,self.sampled_Y)
        filename = './Model/ModelTransfer/RF_'+self.mode+'.sav'
        pickle.dump(rgr, open(filename, 'wb'))       
        
    def SVR(self,
           X=None,
           Y=None):
        """This function is used to implement a Support Vector Regressor
        """
        from sklearn.svm import SVR 
        rgr = SVR(gamma='scale', C=1.0, epsilon=0.1)
        if (X is not None and 
            Y is not None):
            (self.sampled_X,self.sampled_Y)=(X,Y)
        # train    
        rgr.fit(self.sampled_X,self.sampled_Y)
        
        filename = './Model/ModelTransfer/SVR_'+self.mode+'.sav'
        pickle.dump(rgr, open(filename, 'wb')) 
        
    def BRT(self,
           X=None,
           Y=None):
        """This function is used to implement boosted regression tree 
        """    
        from sklearn.ensemble import GradientBoostingRegressor
        rgr=GradientBoostingRegressor()
        if (X is not None and 
            Y is not None):
            (self.sampled_X,self.sampled_Y)=(X,Y)
        # train    
        
        rgr.fit(self.sampled_X,self.sampled_Y)
        filename = './Model/ModelTransfer/BRT_'+self.mode+'.sav'
        pickle.dump(rgr, open(filename, 'wb')) 
        
        
    def NN(self,
           X=None,
           Y=None):
        """This function is used to implement Neural Network
        """
        from sklearn.neural_network import MLPRegressor
        rgr=MLPRegressor(hidden_layer_sizes=(40,),
                         activation="tanh",
                         solver="lbfgs")
        if (X is not None and 
            Y is not None):
            (self.sampled_X,self.sampled_Y)=(X,Y)
        # train    
        
        rgr.fit(self.sampled_X,self.sampled_Y)
        filename = './Model/ModelTransfer/NN_'+self.mode+'.sav'
        pickle.dump(rgr, open(filename, 'wb'))
          
    def MARS(self,
            X=None,
            Y=None):
        """This function is used to imeplement Multivariate Adadptive Regression Splines
        """
        from pyearth import Earth
        rgr=Earth()
        if (X is not None and 
            Y is not None):
            (self.sampled_X,self.sampled_Y)=(X,Y)
        # train    
        rgr.fit(self.sampled_X,self.sampled_Y)
        rgr.fit(self.sampled_X,self.sampled_Y)
        filename = './Model/ModelTransfer/MARS_'+self.mode+'.sav'
        pickle.dump(rgr, open(filename, 'wb')) 
        
        
                
        
        
