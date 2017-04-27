from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import  pickle,os
from regressionindex import RegressionIndex
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import average_precision_score

class Utils():
    def __init__(self):
        self.Kfold = "skfold_1006_br.dat"
    
    def get_current_time(self):
        import time
        return time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    
    
    def get_precision_and_recall_f1(self,class_origin_l, class_predict_l):
        #print(class_predict_l.shape[0])
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
       
        for i in range(class_predict_l.shape[0]):
            #print(i)
            if class_predict_l[i] == class_origin_l[i]:
                if class_predict_l[i] == 1:
                    tp = tp + 1
                else:
                    tn = tn + 1
            else:
                if class_predict_l[i] == 1:
                    fp = fp + 1
                else:
                    fn = fn + 1
        acc = (tp+tn)/(tp+tn+fp+fn)
        if tp + fp == 0:
            precision  = 0.0
        else:
            precision = tp/(tp+fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp/(tp+fn)
        
        if tp == 0:
            f1 = 0.0
        else:
            f1 = 2*precision*recall/(precision+recall)
        return precision, recall, f1, acc
        
        
        
   
            
                   
    def calc_auc(self, cls, class_predict, num_class = 1):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        
        for i in range(num_class):
           
            fpr[i], tpr[i], _ = roc_curve(cls[:, i], class_predict[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
        y_origin_r = cls.reshape(-1, 1)
        y_predict_r = class_predict.reshape(-1, 1)  
        
         
        fpr['micro'], tpr['micro'], _ = roc_curve(y_origin_r, y_predict_r)
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        return roc_auc[1]  
    
    
    def calc_auc_t(self, cls, class_predict):
        
        fpr, tpr, _ = roc_curve(cls, class_predict)
        roc_auc = auc(fpr, tpr)
        
        pr_auc = average_precision_score(cls, class_predict)  
          
        return roc_auc,pr_auc
    
    
        
    def save_result(self, class_origin, class_predict, file_="deeph"):
        curr_t = self.get_current_time()
        fiw1 = open('results/' + file_ + '_' + curr_t + '_class_origin.txt', 'w')
        fiw2 = open('results/' + file_ + '_' + curr_t + '_class_predict.txt', 'w')
        for s in class_origin:
            for i in s:
                fiw1.write(str(i) + ' ')
            fiw1.write('\n')
        fiw1.close()
        for s in class_predict:
            for i in s:
                fiw2.write(str(i) + ' ')
            fiw2.write('\n ')
        fiw2.close()

                  

    
        
    def show_each_cancer_details(self, cls, class_predict, cancer_id=-1, name="FCN"):
        # -1 all cancers
        cancer_start_pos = [0, 404, 1489, 1781, 1814, 2186, 2232, 2416, 2580, 3099, 3163, 3694, 3981, 4130, 4654, 5021, 5525, 6018, 6324, 6501, 6998, 7260, 7720, 8107, 8245, 8752, 8870, 9240, 9308]
        if cancer_id == -1:
            cls_id = cls
            class_predict_id = class_predict
        else:
            start = cancer_start_pos[cancer_id-1]
            end = cancer_start_pos[cancer_id]
            cls_id = cls[start:end]
            class_predict_id = class_predict[start:end]
        
        
        print(len(cls_id))
        ri = RegressionIndex()
        mae = ri.calc_MAE(cls_id, class_predict_id)
        mse = ri.calc_MSE(cls_id, class_predict_id)
        rmse = ri.calc_RMSE(cls_id, class_predict_id)
        nrmse = ri.calc_NRMSE(cls_id, class_predict_id)
        r2 = ri.calc_R_square(cls_id, class_predict_id)
                
        print("MAE from " + name + " : " + str(mae))
        print("MSE from " + name + " : " + str(mse))
        print("RMSE from " + name + " : " + str(rmse))
        print("NRMSE from " + name + " : " + str(nrmse))
        print("R2 from " + name + " : " + str(r2))
        
        
    def code_lables(self, d_class, num_class):
    # #[1,2]  -->  [1,0][0,1]
        coding = []
        cls = []
        labels = np.array(np.zeros(len(d_class)))
        j = -1 
        for row in d_class:
            j = j + 1
            labels[j] = row
            for i in range(num_class):
            # for i in [1,7]:
                if row == i:
                    coding.append(1)
                else:
                    coding.append(0)
            
            # print(coding)
     
            cls.append(coding)
            coding = []
        cls = np.array(cls).astype(float)
        return labels, cls   


 



