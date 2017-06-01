###############################
## An implementation of multimodal deep neural network, a new model for human breast cancer prognosis prediction.
## Version 1.0
###############################

#numpy should add first
import numpy
import numpy as np
import tensorflow as tf
import random, os, math
import  pickle
from utils import Utils
from sklearn.cross_validation import StratifiedKFold
import ConfigParser
from numpy import float32

class MDNNMD():
    def __init__(self):
        self.name = 'MDNNMD'
        self.K = 10
        self.D1 = "Expr-400"
        self.D2 = 'CNA-200'
        self.D3 = 'CLINICAl-25'
        self.alpha = 0.4
        self.beta = 0.1
        self.gamma = 0.6
        self.LABEL = 'os_label_1980'
        self.Kfold = "data/METABRIC_5year_skfold_1980_491.dat"
        self.epsilon = 1e-3
        self.BATCH_SIZE = 128
        self.END_LEARNING_RATE = 0.001
        self.F_SIZE = 400
        self.hidden_units = [3000,3000,3000,100] 
        self.MT_CLASS_TASK1 = 2
        self.IS_PT = "F"
        self.MODEL = dict()
        self.IS_PRINT_INFO = "T"
        self.TRAINING = "True"
        self.active_fun = 'tanh'
        self.MAX_STEPS = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000] 
       
       
    def load_config(self):
        cp = ConfigParser.SafeConfigParser()
        cp.read('mdnnmd.conf')
        self.alpha = float32(cp.get('input', 'alpha'))
        self.beta = float32(cp.get('input', 'beta'))
        self.gamma = float32(cp.get('input', 'gamma'))
        self.D1 = cp.get('input', 'D1')
        self.D2 = cp.get('input', 'D2')
        self.D3 = cp.get('input', 'D3')
        self.K = int(cp.get('input', 'K'))
        self.LABEL = cp.get('input', 'label')
          
          
        self.BATCH_SIZE = int(cp.get('dnn', 'batch_size'))
        self.epsilon = float32(cp.get('dnn', 'bne'))
        self.active_fun =  cp.get('dnn', 'active_function')
        
      
    def scale_max_min(self, data, lower=-1, upper=1):
        max_value = np.max(np.max(data, 0),0)  
        min_value = np.min(np.min(data, 0),0)
        r = np.size(data, 0)
        c = np.size(data, 1)
        for i in range(r):
            for j in range(c):
                data[i,j] = lower + (upper-lower)*((data[i,j]-min_value)/(max_value-min_value))
        return data
         
    def next_batch(self,train_f,train_l1,batch_size,i):
        num = int((train_f.shape[0])/batch_size-1)
        i = i%num
        train_indc = range(train_f.shape[0])
        if i == num-1:
            random.shuffle(train_indc) 
        xs = train_f[train_indc[i*batch_size:(i+1)*batch_size]]
        y1 = train_l1[train_indc[i*batch_size:(i+1)*batch_size]]
       
        return xs,y1
  
    def batch_norm_wrapper(self,inputs,is_training,decay = 0.999):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
                           
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        
        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1])
            
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            
            with tf.control_dependencies([train_mean,train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, self.epsilon)
            
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, self.epsilon) 
   
        
        
    def code_lables(self, d_class, num_class):
    # #[1,2]  -->  [1,0][0,1]
        coding = []
        cls = []
        labels = numpy.array(numpy.zeros(len(d_class)))
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
            cls.append(coding)
            coding = []
        cls = numpy.array(cls).astype(float)
        return labels, cls   
    
    def packaging_model(self, weight1, biase1, weight2, biase2, Y1_weight, Y1_biase):
        model = dict()
        model["weight1"] = weight1
        model["biase1"] = biase1
        model["weight2"] = weight2
        model["biase2"] = biase2
        model["Y1_weight"] = Y1_weight
        model["Y1_biase"] = Y1_biase
        return model


    
    def train(self, kf1,d_matrix, d_class, cls, ut):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, self.F_SIZE], name='x-input')  
            y1_ = tf.placeholder(tf.float32, [None, self.MT_CLASS_TASK1], name='y-input')
            keep_prob = tf.placeholder(tf.float32)
            f_gene_exp = x
          
        with tf.name_scope('hidden1'):  
            weight1 = tf.Variable(tf.truncated_normal([self.F_SIZE, self.hidden_units[0]], stddev=1.0 / math.sqrt(float(self.F_SIZE)/2), seed = 1,name='weights'))     
            biase1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[0]]))        
            hidden1_mu = tf.matmul(f_gene_exp, weight1) + biase1
            hidden1_BN = self.batch_norm_wrapper(hidden1_mu,  self.TRAINING)
            
            if self.active_fun == 'relu':
                hidden1 = tf.nn.relu(hidden1_BN )
            else:
                hidden1 = tf.nn.tanh(hidden1_BN )
            
        with tf.name_scope('hidden2'):
            weight2 = tf.Variable(tf.truncated_normal([self.hidden_units[0], self.hidden_units[1]], stddev=1.0 / math.sqrt(float(self.hidden_units[0])/2),  seed = 1,name='weights'))
            biase2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[1]]))
            hidden2_mu = tf.matmul(hidden1, weight2) + biase2
            hidden2_BN = self.batch_norm_wrapper(hidden2_mu,  self.TRAINING)
            hidden2 = tf.nn.tanh(hidden2_BN)
            
            if self.active_fun == 'relu':
                hidden2 = tf.nn.relu(hidden2_BN)
            else:
                hidden2 = tf.nn.tanh(hidden2_BN)
   
        with tf.name_scope('hidden3'):
            weight3 = tf.Variable(tf.truncated_normal([self.hidden_units[1], self.hidden_units[2]], stddev=1.0 / math.sqrt(float(self.hidden_units[1])/2),  seed = 1,name='weights'))
            biase3 = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[2]])) 
            hidden3_mu = tf.matmul(hidden2, weight3) + biase3
            hidden3_BN = self.batch_norm_wrapper(hidden3_mu,  self.TRAINING)
            hidden3 = tf.nn.tanh(hidden3_BN)
            if self.active_fun == 'relu':
                hidden3 = tf.nn.relu(hidden3_BN)
            else:
                hidden3 = tf.nn.tanh(hidden3_BN)
       
        # # dropout layer
        with tf.name_scope('dcl1'):         
            Y1_weight = tf.Variable(tf.truncated_normal([self.hidden_units[2], self.hidden_units[3]], stddev=1.0 / math.sqrt(float(self.hidden_units[2])/2),  seed = 1,name='weights'))
            Y1_biase = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[3]]))
            Y1_h_dc1_mu = tf.matmul(hidden3, Y1_weight) + Y1_biase
            Y1_h_dc1_BN = self.batch_norm_wrapper(Y1_h_dc1_mu,  self.TRAINING)
            
            if self.active_fun == 'relu':
                Y1_h_dc1_drop  = tf.nn.relu(Y1_h_dc1_BN)
            else:
                Y1_h_dc1_drop  = tf.nn.tanh(Y1_h_dc1_BN)
             
            Y1_h_dc1_drop_c = Y1_h_dc1_drop
                                    
        with tf.name_scope('full_connected'):
            Y1_weight_fc1 = tf.Variable(tf.truncated_normal([self.hidden_units[3], self.MT_CLASS_TASK1], stddev=1.0 / math.sqrt(float(self.hidden_units[3])/2), seed = 1, name='weight-Y1-fc'))          
            Y1_biase_fc1 = tf.Variable(tf.constant(0.1, shape=[self.MT_CLASS_TASK1]))
           
            Y1_pre = (tf.matmul(Y1_h_dc1_drop_c, Y1_weight_fc1) + Y1_biase_fc1)
            Y1 = tf.nn.softmax(Y1_pre)
            

        with tf.name_scope('cross_entropy'):
            Y1_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y1_, logits = Y1_pre))
            Joint_loss = Y1_cross_entropy 
        
        with tf.name_scope('training'):
         
            train_step = tf.train.AdamOptimizer(self.END_LEARNING_RATE).minimize(Joint_loss)     
    
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(Y1, 1), tf.argmax(y1_, 1))
                
                
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        def run_CV(train_f,train_l1,test_f,test_l1,i_k):
            
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            sess = tf.InteractiveSession(config=config)
            tf.global_variables_initializer().run()
    
            def feed_dict(train,i):
                if train:
                    batch_size = self.BATCH_SIZE
                    xs, y1 = self.next_batch(train_f,train_l1,batch_size,i)
                    k = 1.0
                else:
                    xs, y1 = test_f,test_l1
                    k = 1.0
                return {x: xs, y1_: y1, keep_prob: k}
            
        
            for i in range(1,self.MAX_STEPS[i_k-1]+1):
                    
                self.TRAINING = "True"
                result = train_step.run(feed_dict=feed_dict(True,i))    
                
                 
            self.TRAINING = "False"
            test_Y1 = Y1.eval(feed_dict=feed_dict(False,i))       
            sess.close()   
            return test_Y1

        class_predict_fcn_t = numpy.zeros([d_matrix.shape[0], self.MT_CLASS_TASK1])

        i = 0
        for train_indc, test_indc in kf1: 
      
            i += 1       
            print('K fold: %s' % (i)) 
               
            class_predict_fcn_t[test_indc]= run_CV(d_matrix[train_indc], cls[train_indc], 
                                                 d_matrix[test_indc], 
                                                    cls[test_indc],i)
       
        return class_predict_fcn_t

                       
    def load_txt(self,op):     
        d_class = numpy.loadtxt(self.LABEL, delimiter=' ').reshape(-1, 1) 
        d_matrix = numpy.loadtxt(op, delimiter=' ')
        self.F_SIZE = d_matrix.shape[1]
                         
        return d_matrix, d_class
    

ut = Utils() 
#Expr-400
dnn_md1 = MDNNMD()
dnn_md1.load_config()
d_matrix, d_class = dnn_md1.load_txt(dnn_md1.D1)
dnn_md1.MAX_STEPS = [40,30,40,40,40,40,45,60,75,50]   #3000,3000,3000,100 MRMR-400  0504
dnn_md1.hidden_units = [3000,3000,3000,100]
#dnn_md1.active_fun = 'relu'
dnn_md1.IS_PRINT_INFO = "F"
label1, cls = dnn_md1.code_lables(d_class, dnn_md1.MT_CLASS_TASK1)

if os.path.exists(dnn_md1.Kfold):
    kf1 = pickle.load(open(dnn_md1.Kfold,"rb"))
    print("successfully loading already existing kfold index!")
else:
    kf1 = StratifiedKFold(label1, n_folds=dnn_md1.K)
    pickle.dump(kf1, open(dnn_md1.Kfold, "wb"))
    print("successfully generating kfold index!")
class_predict_fcn1 = dnn_md1.train(kf1,d_matrix, d_class, cls, ut)

#CNA
dnn_md2 = MDNNMD() 
dnn_md2.load_config()
d_matrix, d_class = dnn_md2.load_txt(dnn_md2.D2)
dnn_md2.MAX_STEPS = [25,30,25,35,40,45,45,70,85,25]   #3000,3000,3000,100 CNV-200  0504
dnn_md2.hidden_units = [3000,3000,3000,100]
#dnn_md2.active_fun = 'relu'
dnn_md2.IS_PRINT_INFO = "F"
label1, cls = dnn_md2.code_lables(d_class, dnn_md2.MT_CLASS_TASK1)
class_predict_fcn2 = dnn_md2.train(kf1,d_matrix, d_class, cls, ut)

#CLINICAL-25
dnn_md3 = MDNNMD() 
dnn_md3.load_config()
d_matrix, d_class = dnn_md3.load_txt(dnn_md3.D3)
dnn_md3.MAX_STEPS = [365,600,600,600,600,600,430,500,500,500]   #3000,3000,3000,100 CLINICAL-25
dnn_md3.hidden_units = [3000,3000,3000,100]
#dnn_md3.active_fun = 'relu'
dnn_md3.IS_PRINT_INFO = "F"
label1, cls = dnn_md3.code_lables(d_class, dnn_md3.MT_CLASS_TASK1)

class_predict_fcn3 = dnn_md3.train(kf1,d_matrix, d_class, cls, ut)
class_predict_fcn = dnn_md1.alpha*class_predict_fcn1 + dnn_md1.beta*class_predict_fcn2 + dnn_md1.gamma*class_predict_fcn3
## DNN
auc_fcn, pr_auc_fcn = ut.calc_auc_t(cls[:,1], class_predict_fcn[:,1])

pre_f,rec_f,f1_f,acc_f = ut.get_precision_and_recall_f1(np.argmax(cls,1), np.argmax(class_predict_fcn,1))

print("DNN## ACC: %s,AUC %s,PRE %s,REC %s,F1 %s, PR_AUC %s" %(acc_f,auc_fcn,pre_f,rec_f,f1_f,pr_auc_fcn))

                              
name = dnn_md1.name+'_'+str(dnn_md1.hidden_units[0])+'-'+str(dnn_md1.hidden_units[1])+'-'+str(dnn_md1.hidden_units[2])+'-'+str(dnn_md1.hidden_units[3])+'_'+str(dnn_md1.alpha)+'_'+str(dnn_md1.beta)
## save results
#LABEL
np.savetxt("Prediction_labels.txt", np.argmax(class_predict_fcn,1))

#SCORE

np.savetxt("Prediction_score.txt", class_predict_fcn[:,1])

