import numpy as np 
class RegressionIndex():
    """
    Attributes
    ----------
    y_ : numpy.array.
        the original value of samples
    
    y : numpy.array.
        the predict value of samples
        
    References
    ----------
    A Class implements Some Common Regression Index. 
    Copyright 2016 HI-lab. All Rights Reserved.
    Author: Dongdong Sun     Date: 2016-10-25
    
    """
    def __init__(self):
        pass
    
        
    def calc_MAE(self, y_, y):
        # mean absolute error
        
        mae = np.sum(np.abs(y - y_)) / len(y)
        return mae
    
    def calc_MSE(self, y_, y):
        mse = np.sum(np.power((y - y_), 2)) / len(y)
        return mse
    
    def calc_RMSE(self, y_, y):
        # root mean squared error
        
        rmse = np.power(np.sum(np.power((y - y_), 2)) / len(y),0.5)
        return rmse
    
    def calc_NRMSE(self, y_, y):
        # normalixed root mean squared error (deviation) (NRMSE or NRMSD)
        d_v = max(y_)-min(y_)
        nrmse = self.calc_RMSE(y_, y)/d_v
        return nrmse
    
    def calc_CV_RMSE(self, y_, y):
        # coefficient of variation of the RMSE (RMSD)
        mean_y_ = np.sum(y_)/len(y_)
        cv_rmse = self.calc_RMSE(y_, y)/mean_y_
        return cv_rmse
    
    def calc_SSR(self, y_, y):
        # sum of squares of the regression
        mean_y_ = np.sum(y_)/len(y_)
        ssr = np.sum(np.power((y - mean_y_), 2))
        return ssr
    
    def calc_SST(self, y_, y):
        mean_y_ = np.sum(y_)/len(y_)
        sst = np.sum(np.power((y_ - mean_y_), 2))
        return sst
    
    def calc_SSE(self, y_, y):
        sse = np.sum(np.power((y_ - y), 2))
        return sse
        
        
    def calc_R_square(self, y_, y):
        sse = self.calc_SSE(y_, y)
        sst = self.calc_SST(y_, y)
        R_square = 1-sse/sst
        return R_square
        
        
        
    
    