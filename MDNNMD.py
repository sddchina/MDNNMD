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
from sklearn.cross_validation import train_test_split

class MDNNMD():
    def __init__(self):
        self.name = 'MDNNMD'
        self.K = 10
        self.D1 = "Expr-400"
        self.D2 = 'CNA-200'
        self.D3 = 'CLINICAl-25'
        self.alpha = 0.4
        self.beta = 0.1
        self.gamma = 0.5
        self.LABEL = 'os_label_1980'
        self.Kfold = "data/METABRIC_5year_skfold_1980_491.dat"
        self.epsilon = 1e-3
        self.BATCH_SIZE = 64
        self.END_LEARNING_RATE = 0.001
        self.F_SIZE = 400
        self.hidden_units = [3000,3000,3000,100] 
        self.MT_CLASS_TASK1 = 2
        self.IS_PT = "F"
        self.MODEL = dict()
        self.IS_PRINT_INFO = "T"
        self.TRAINING = "True"
        self.active_fun = 'tanh'
        self.drop = 0.5
        self.regular = True
        self.lrd = False
        self.curr_fold = 1
        self.MAX_STEPS = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000] 
        self.epoch = 100
       
       
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
        
      
    def scale_max_min(self, data, lower=0, upper=1):
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
            
            
            if self.active_fun == 'relu':
                hidden2 = tf.nn.relu(hidden2_BN)
            else:
                hidden2 = tf.nn.tanh(hidden2_BN)
   
        with tf.name_scope('hidden3'):
            weight3 = tf.Variable(tf.truncated_normal([self.hidden_units[1], self.hidden_units[2]], stddev=1.0 / math.sqrt(float(self.hidden_units[1])/2),  seed = 1,name='weights'))
            biase3 = tf.Variable(tf.constant(0.1, shape=[self.hidden_units[2]])) 
            hidden3_mu = tf.matmul(hidden2, weight3) + biase3
            hidden3_BN = self.batch_norm_wrapper(hidden3_mu,  self.TRAINING)
            
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
             
            Y1_h_dc1_drop_c = tf.nn.dropout(Y1_h_dc1_drop, keep_prob)
                                    
        with tf.name_scope('full_connected'):
            Y1_weight_fc1 = tf.Variable(tf.truncated_normal([self.hidden_units[3], self.MT_CLASS_TASK1], stddev=1.0 / math.sqrt(float(self.hidden_units[3])/2), seed = 1, name='weight-Y1-fc'))          
            Y1_biase_fc1 = tf.Variable(tf.constant(0.1, shape=[self.MT_CLASS_TASK1]))
           
            Y1_pre = (tf.matmul(Y1_h_dc1_drop_c, Y1_weight_fc1) + Y1_biase_fc1)
            Y1 = tf.nn.softmax(Y1_pre)
            

        with tf.name_scope('cross_entropy'):
            l2_loss = 0
            if self.regular:
                l2_loss = tf.nn.l2_loss(weight1) + tf.nn.l2_loss(weight2) + tf.nn.l2_loss(weight3)  + tf.nn.l2_loss(Y1_weight) + tf.nn.l2_loss(Y1_weight_fc1) 
                beta = 1e-4
                l2_loss *= beta
            Y1_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y1_, logits = Y1_pre))
            Joint_loss = Y1_cross_entropy + l2_loss
            
            
            #train_loss = tf.summary.scalar('train_loss', Joint_loss)
            #valid_loss = tf.summary.scalar('valid_loss', Joint_loss)
            #######
            valid_loss = tf.summary.scalar('train_loss', Joint_loss)
            train_loss = tf.summary.scalar('valid_loss', Joint_loss)
        
        with tf.name_scope('training'):
            if self.lrd:
                cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
                starter_learning_rate = 0.4
                learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 100000, 0.96, staircase=True)
                train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(Joint_loss, global_step=cur_step)
            else:
                train_step = tf.train.AdamOptimizer(self.END_LEARNING_RATE).minimize(Joint_loss)     
    
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(Y1, 1), tf.argmax(y1_, 1))
                
                
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
               
                
        def run_CV(train_f,train_l,vali_f,vali_l,test_f,test_l,i_k):
            
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            sess = tf.InteractiveSession(config=config)
            train_writer = tf.summary.FileWriter('logs/train',sess.graph)
            valid_writer = tf.summary.FileWriter('logs/valid')
            tf.global_variables_initializer().run()
    
            def feed_dict(train,validation,i):
                if train:
                    batch_size = self.BATCH_SIZE
                    xs, y1 = self.next_batch(train_f,train_l,batch_size,i)
                    k = self.drop
                elif validation:
                    xs, y1 = vali_f,vali_l
                    k = 1.0                 
                else:
                    xs, y1 = test_f,test_l
                    k = 1.0
                return {x: xs, y1_: y1, keep_prob: k}
            
            
            epochs = -1
            for i in range(0,self.MAX_STEPS[i_k-1]*10+1):
                self.TRAINING = "True"
                _,loss,Y1_cross_entropy_my= sess.run([train_step,train_loss,Y1_cross_entropy],feed_dict=feed_dict(True,False,i))   
                
                #print(Y1_cross_entropy_my)
               
                if i % 10 == 0:
                    epochs = epochs+1
                    #if self.curr_fold == 3:
                    train_writer.add_summary(loss, epochs)
                    self.TRAINING = "False"
                    #if self.curr_fold == 3:
                    valid_writer.add_summary(valid_loss.eval(feed_dict=feed_dict(False,True,i)),epochs)
                    test_Y1 = Y1.eval(feed_dict=feed_dict(False,False,i)) 
                    JS = Joint_loss.eval(feed_dict=feed_dict(False,False,i))
                    auc1, pr_auc1 = ut.calc_auc_t(test_l[:,1], test_Y1[:,1])
                    pre,rec,f1,acc = ut.get_precision_and_recall_f1(np.argmax(test_l,1), np.argmax(test_Y1,1))
                    if self.IS_PRINT_INFO == "T":
                        print('Accuracy at step %s: accT:%3f auc1:%f  precision:%3f recall:%3f  f1:%3f' % (i,acc, auc1,pre,rec,f1))
                    
                
                 
            self.TRAINING = "False"
            test_Y1 = Y1.eval(feed_dict=feed_dict(False,False,i)) 
            valid_Y1 = Y1.eval(feed_dict=feed_dict(False,True,i)) 
            train_writer.close()
            valid_writer.close()     
            sess.close()   
            return test_Y1,valid_Y1

        class_predict_fcn_t = numpy.zeros([d_matrix.shape[0], self.MT_CLASS_TASK1])
        
    
        i = 0
        for train_indc, test_indc in kf1: 
      
            i += 1       
            
            if i == 2:
                exit()
            
            print('K fold: %s' % (i)) 
               
            X_train, X_valid, y_train, y_valid = train_test_split(d_matrix[train_indc], d_class[train_indc], test_size=0.2, random_state=0)
            X_test =  d_matrix[test_indc]
            y_test =  cls[test_indc]
            
            label11, cls_train = self.code_lables(y_train, self.MT_CLASS_TASK1)
            label12, cls_valid = self.code_lables(y_valid, self.MT_CLASS_TASK1)
            
            class_predict_fcn_t[test_indc], valid_Y2= run_CV(X_train, cls_train, X_valid, cls_valid, X_test, y_test,i)
            if i == 1:
                cls_valid_all = cls_valid
                p_valid_all = valid_Y2
            else:
                cls_valid_all = np.vstack((cls_valid_all,cls_valid))
                p_valid_all = np.vstack((p_valid_all,valid_Y2))
            
        return class_predict_fcn_t,p_valid_all,cls_valid_all

                       
    def load_txt(self,op,f_len):     
        
        d_class = numpy.loadtxt(self.LABEL, delimiter=' ').reshape(-1, 1) 
        d_matrix = numpy.loadtxt(op, delimiter=' ')
        
        d_matrix = d_matrix[:,0:f_len]
        self.F_SIZE = d_matrix.shape[1]
                         
        return d_matrix, d_class
    

ut = Utils() 
#Expr-400
dnn_md1 = MDNNMD()
dnn_md1.load_config()
d_matrix, d_class = dnn_md1.load_txt(dnn_md1.D1,400)
dnn_md1.epoch = 40
dnn_md1.MAX_STEPS = [dnn_md1.epoch,dnn_md1.epoch,dnn_md1.epoch,dnn_md1.epoch,dnn_md1.epoch,dnn_md1.epoch,dnn_md1.epoch,dnn_md1.epoch,dnn_md1.epoch,dnn_md1.epoch]   #3000,3000,3000,100 MRMR-400  0504
#dnn_md1.MAX_STEPS = [50,50,50,50,50,50,50,50,50,50]
dnn_md1.hidden_units = [1000,500,500,100]
##dnn_md1.active_fun = 'relu'
#dnn_md1.regular = 'True'
dnn_md1.END_LEARNING_RATE = 0.00001
dnn_md1.IS_PRINT_INFO = "F"
label1, cls = dnn_md1.code_lables(d_class, dnn_md1.MT_CLASS_TASK1)
# 
if os.path.exists(dnn_md1.Kfold):
    kf1 = pickle.load(open(dnn_md1.Kfold,"rb"))
    print("successfully loading already existing kfold index!")
else:
    kf1 = StratifiedKFold(label1, n_folds=dnn_md1.K)
    pickle.dump(kf1, open(dnn_md1.Kfold, "wb"))
    print("successfully generating kfold index!")
#class_predict_fcn1,p_valid_all1,cls_valid_all1 = dnn_md1.train(kf1,d_matrix, d_class, cls, ut)




#CNA
dnn_md2 = MDNNMD() 
dnn_md2.load_config()
d_matrix, d_class = dnn_md2.load_txt(dnn_md2.D2,200)
#dnn_md2.MAX_STEPS = [25,30,25,35,40,45,45,70,85,25]   #3000,3000,3000,100 CNV-200  0504
dnn_md2.epoch = 40
dnn_md2.MAX_STEPS = [dnn_md2.epoch,dnn_md2.epoch,dnn_md2.epoch,dnn_md2.epoch,dnn_md2.epoch,dnn_md2.epoch,dnn_md2.epoch,dnn_md2.epoch,dnn_md2.epoch,dnn_md2.epoch] 
dnn_md2.hidden_units = [1000,500,500,100]
#dnn_md2.active_fun = 'relu'
dnn_md2.END_LEARNING_RATE = 0.00001
dnn_md2.IS_PRINT_INFO = "F" 
label1, cls = dnn_md2.code_lables(d_class, dnn_md2.MT_CLASS_TASK1)
#class_predict_fcn2,p_valid_all2,cls_valid_all2 = dnn_md2.train(kf1,d_matrix, d_class, cls, ut)





#CLINICAL-25
dnn_md3 = MDNNMD() 
dnn_md3.load_config()
d_matrix, d_class = dnn_md3.load_txt(dnn_md3.D3,25)
dnn_md3.epoch = 60
dnn_md3.MAX_STEPS = [dnn_md3.epoch,dnn_md3.epoch,dnn_md3.epoch,dnn_md3.epoch,dnn_md3.epoch,dnn_md3.epoch,dnn_md3.epoch,dnn_md3.epoch,dnn_md3.epoch,dnn_md3.epoch]   #3000,3000,3000,100 CLINICAL-25
dnn_md3.hidden_units = [1000,1000,1000,100]  #1000 1000 1000 100
#dnn_md3.active_fun = 'relu'
dnn_md3.BATCH_SIZE = 128
dnn_md3.END_LEARNING_RATE = 0.001
dnn_md3.drop = 1.0
dnn_md3.IS_PRINT_INFO = "F"
label1, cls = dnn_md3.code_lables(d_class, dnn_md3.MT_CLASS_TASK1)
class_predict_fcn3,p_valid_all3,cls_valid_all3 = dnn_md3.train(kf1,dnn_md3.scale_max_min(d_matrix), d_class, cls, ut)




dnn_md1.alpha = 0.3
dnn_md1.beta = 0.1
dnn_md1.gamma = 1-dnn_md1.alpha-dnn_md1.beta

#validation
#p_valid_all = dnn_md1.alpha*p_valid_all1 + dnn_md1.beta*p_valid_all2 + dnn_md1.gamma*p_valid_all3
#auc_fcn, pr_auc_fcn = ut.calc_auc_t(cls_valid_all1[:,1], p_valid_all[:,1])

#pre_f,rec_f,f1_f,acc_f = ut.get_precision_and_recall_f1(np.argmax(cls_valid_all1,1), np.argmax(p_valid_all,1))

#print("DNN-validation## ACC: %s,AUC %s,PRE %s,REC %s,F1 %s, PR_AUC %s" %(acc_f,auc_fcn,pre_f,rec_f,f1_f,pr_auc_fcn))

#test 
#class_predict_fcn = dnn_md1.alpha*class_predict_fcn1 + dnn_md1.beta*class_predict_fcn2 + dnn_md1.gamma*class_predict_fcn3
#auc_fcn, pr_auc_fcn = ut.calc_auc_t(cls[:,1], class_predict_fcn[:,1])

#pre_f,rec_f,f1_f,acc_f = ut.get_precision_and_recall_f1(np.argmax(cls,1), np.argmax(class_predict_fcn,1))

#print("DNN-testing## ACC: %s,AUC %s,PRE %s,REC %s,F1 %s, PR_AUC %s" %(acc_f,auc_fcn,pre_f,rec_f,f1_f,pr_auc_fcn))

            
name = dnn_md1.name+'_'+str(dnn_md1.hidden_units[0])+'-'+str(dnn_md1.hidden_units[1])+'-'+str(dnn_md1.hidden_units[2])+'-'+str(dnn_md1.hidden_units[3])+'_'+str(dnn_md1.alpha)+'_'+str(dnn_md1.beta)
#SCORE
#np.savetxt("results/result_METABRIC1/p_valid_all1.txt", p_valid_all1)
#np.savetxt("results/result_METABRIC1/p_valid_all2.txt", p_valid_all2)
#np.savetxt("results/result_METABRIC1/p_valid_all3.txt", p_valid_all3)

#np.savetxt("results/result_METABRIC1/cls_valid_all1.txt", cls_valid_all1)

#np.savetxt("results/result_METABRIC1/class_predict_fcn1.txt", class_predict_fcn1)
#np.savetxt("results/result_METABRIC1/class_predict_fcn2.txt", class_predict_fcn2)
#np.savetxt("results/result_METABRIC1/class_predict_fcn3.txt", class_predict_fcn3)
#np.savetxt("results/result_METABRIC1/cls.txt", cls)
