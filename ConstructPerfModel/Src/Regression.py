from Metric import Metric

class Regression(object):
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
        
        if sampled_data_index is not None:
        # Sampled Data
            self.sampled_data_index=sampled_data_index
            (self.sampled_X, self.sampled_Y)=([_ for _ in xrange(len(self.sampled_data_index))],[_ for _ in xrange(len(self.sampled_data_index))])
        
            for i in xrange(len(self.sampled_X)):    
                self.sampled_X[i]=self.X[i]
                self.sampled_Y[i]=self.Y[i]
        
    def get_sampled_data(self):
        """This function is used to return sampled data
        @returns sampled_X, sampled_Y
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
        # test
        Y_pred=rgr.predict(self.X)
        # compute metric
        m_nmse=self.metric.normalized_mean_square_error(Y_pred,self.Y)
        m_mape=self.metric.mean_absolute_percentage_error(Y_pred,self.Y)        
        return (m_nmse,m_mape)
    
    def RF(self,
           X=None,
           Y=None):
        """This function is used to implement a random forest
        """
        from sklearn.ensemble import RandomForestRegressor
        rgr=RandomForestRegressor(n_estimators=512, 
                                  max_depth=None,
                                  max_features=None,
                                  max_leaf_nodes=None)
        if (X is not None and 
            Y is not None):
            (self.sampled_X,self.sampled_Y)=(X,Y)
        # train    
        rgr.fit(self.sampled_X,self.sampled_Y)
        # test
        Y_pred=rgr.predict(self.X)
        # compute metric
        m_nmse=self.metric.normalized_mean_square_error(Y_pred,self.Y)
        m_mape=self.metric.mean_absolute_percentage_error(Y_pred,self.Y)        
        return (m_nmse,m_mape)
        
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
        # test
        Y_pred=rgr.predict(self.X)
        # compute metric
        m_nmse=self.metric.normalized_mean_square_error(Y_pred,self.Y)
        m_mape=self.metric.mean_absolute_percentage_error(Y_pred,self.Y)        
        return (m_nmse,m_mape)
        
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
        print (len(self.sampled_Y))
        # test
        Y_pred=rgr.predict(self.X)
        # compute metric
        m_nmse=self.metric.normalized_mean_square_error(Y_pred,self.Y)
        m_mape=self.metric.mean_absolute_percentage_error(Y_pred,self.Y)        
        return (m_nmse,m_mape)
        
        
    def NN(self,
           X=None,
           Y=None):
        """This function is used to implement Neural Network
        """
        from sklearn.neural_network import MLPRegressor
        import matplotlib.pyplot as plt
        rgr=MLPRegressor(hidden_layer_sizes=(8),
                         activation="tanh",
                         solver="lbfgs")
                         
        if (X is not None and 
            Y is not None):
            (self.sampled_X,self.sampled_Y)=(X,Y)
             
        # train    
        rgr.fit(self.sampled_X,self.sampled_Y)
        # test
        Y_pred=rgr.predict(self.X)
        # compute metric
        m_nmse=self.metric.normalized_mean_square_error(Y_pred,self.Y)
        m_mape=self.metric.mean_absolute_percentage_error(Y_pred,self.Y)        
        return (m_nmse,m_mape)
          
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
        # test
        Y_pred=rgr.predict(self.X)
        # compute metric
        m_nmse=self.metric.normalized_mean_square_error(Y_pred,self.Y)
        m_mape=self.metric.mean_absolute_percentage_error(Y_pred,self.Y)        
        return (m_nmse,m_mape)
        
        
                
        
        
