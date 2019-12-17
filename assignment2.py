# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:15:48 2019

@author: Daniel Eduardo Bravo Solís and Santiago Herrero Melo
"""
import numpy as np
import scipy.io as sc
import os
from dataclasses import dataclass
import math
from sklearn.model_selection import KFold

execution_path = os.getcwd()

"""
# Data used for development validation of the algorithm  
"""

dd = sc.loadmat(os.path.join(execution_path ,"template_data\data/ejemplolineargaussian.mat"))
dd = dd['ejemplo'][0]

test_X = dd['inputX'][0]
test_Y = dd['inputY'][0]
test_betas = dd['outputBetas'][0]
test_sigma = dd['outputSigma'][0]

vd = sc.loadmat(os.path.join(execution_path ,"template_data\data/validation_data.mat"))
test_model_nb = vd['model_nb']  

train_set = vd['train_indexes']==[1]
labels_train = vd['labels_small'][train_set].reshape(-1,1)
data_train = vd['data_small'][:,:,train_set[:,0]]

test_set = vd['test_indexes']==[1]
labels_test = vd['labels_small'][test_set].reshape(-1,1)
data_test = vd['data_small'][:,:,test_set[:,0]]

full_valid_set = vd['data_small']
full_labels_set = vd['labels_small']

# Skeleton definition
NUMBER_OF_CLASSES = 4
NB_MODEL = 0
LG_MODEL = 1

# Skeleton definition
NUI_SKELETON_POSITION_COUNT = 20

NONE = -1
HIP_CENTER = 0
SPINE = 1
SHOULDER_CENTER = 2
HEAD = 3
SHOULDER_LEFT = 4
ELBOW_LEFT = 5
WRIST_LEFT = 6
HAND_LEFT = 7
SHOULDER_RIGHT = 8
ELBOW_RIGHT = 9
WRIST_RIGHT = 10
HAND_RIGHT = 11
HIP_LEFT = 12
KNEE_LEFT = 13
ANKLE_LEFT = 14
FOOT_LEFT = 15
HIP_RIGHT = 16
KNEE_RIGHT = 17
ANKLE_RIGHT = 18
FOOT_RIGHT = 19

nui_skeleton_names = ( \
    'HIP_CENTER', 'SPINE', 'SHOULDER_CENTER', 'HEAD', \
    'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT', 'HAND_LEFT', \
    'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT', 'HAND_RIGHT', \
    'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT', 'FOOT_LEFT', \
    'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT', 'FOOT_RIGHT' )

nui_skeleton_conn = ( \
    NONE, \
    HIP_CENTER, \
    SPINE, \
    SHOULDER_CENTER, \
    # Left arm 
    SHOULDER_CENTER, \
    SHOULDER_LEFT,  \
    ELBOW_LEFT,  \
    WRIST_LEFT,  \
    # Right arm 
    SHOULDER_CENTER,  \
    SHOULDER_RIGHT,  \
    ELBOW_RIGHT,  \
    WRIST_RIGHT,  \
    # Left leg 
    HIP_CENTER,  \
    HIP_LEFT,  \
    KNEE_LEFT,  \
    ANKLE_LEFT,  \
    # Right leg 
    HIP_CENTER,  \
    HIP_RIGHT,  \
    KNEE_RIGHT,  \
    ANKLE_RIGHT,  \
)

""" 
Classes to create the model and its sub-elements 
"""
@dataclass
class jointPartsT:
    j_mean: np.array
    j_sigma: np.array
    j_betas: np.array
    j_nb: int 
    
    def __init__(self, params=None):
        pass

@dataclass
class modelType:
    connectivity: list()
    class_priors: np.array
    jointparts: list()
    
    def __init__(self, params=None):
        pass

def load_dataset(file=None):
    """
      Returns the data, the labels and the person id for each action
    """
    import scipy.io
    
    if file is None:
        ex = scipy.io.loadmat(os.path.join(execution_path , "template_data\data\data.mat"))
    else:
        ex = scipy.io.loadmat(file)
        
    return ex['data'],ex['labels'],ex['individuals']

def fit_gaussian(vector):
    return np.mean(vector),np.std(vector)

def my_cov(x,y,w=None):
    """
      Computes the covarinace given x and y vectors
    """
    if w is None:
        return (np.mean(x*y)-np.mean(x)*np.mean(y))
    else:
        return np.sum(w*x*y)/np.sum(w)-np.sum(w*x)*np.sum(w*y)/np.sum(w)/np.sum(w)

def fit_linear_gaussian(Y,X):
    """
    Input:
      Y: vector of size D with the observations for the variable
      X: matrix DxV with the observations for the parent variables
                 of X. V is the number of parent variables
      W: vector of size D with the weights of the instances (ignore for the moment)
      
    Outout:
       The betas and sigma
    """
    num_instances, cols = X.shape
    X_mod = np.insert(X,0,1,axis=1)
    A = np.zeros((cols+1,cols+1))
    b = np.zeros((cols+1,1))
    for m in range(0, cols+1):
        b[m]= np.mean(np.transpose(Y)*(X_mod[:,m]))
        for n in range(0,cols+1):
            A[m,n]= np.mean(X_mod[:,m]*X_mod[:,n])
            
    betas = np.linalg.solve(A,b)
    cova = 0;
    
    for i in range (0, cols):
        for j in range (0, cols):
            cova = cova + betas[i+1]*betas[j+1]*my_cov(X[:,i],X[:,j]) #Beta in position 0 is not included!

#    sigma = np.sqrt(my_cov(Y,Y)-cova)
    sigma = my_cov(Y,Y)-cova
    return (sigma,betas)


def learn_model(dataset, labels,G=None):
    """
    Input:
     dataset: The data as it is loaded from load_data
     labels:  The labels as loaded from load_data
     Graph:   (optional) If None, this def should compute the naive 
           bayes model. If it contains a skel description (pe 
           nui_skeleton_conn, as obtained from skel_model) then it should
           compute the model using the Linear Gausian Model

    Output: the model
     a (tentative) structure for the output model is:
       model.connectivity: the input Graph variable should be stored here 
                           for later use.
       model.class_priors: containing a vector with the prior estimations
                           for each class
       model.jointparts[i] contains the estimated parameters for the i-th joint

          For joints that only depend on the class model.jointparts(i) has:
            model.jointparts(i).means: a matrix of 3 x #classes with the
                   estimated means for each of the x,y,z variables of the 
                   i-th joint and for each class.
            model.jointparts(i).sigma: a matrix of 3 x #classes with the
                   estimated stadar deviations for each of the x,y,z 
                   variables of the i-th joint and for each class.

          For joints that follow a gausian linear model model.jointparts(i) has:
            model.jointparts(i).betas: a matrix of 12 x #classes with the
                   estimated betas for each x,y,z variables (12 in total) 
                   of the i-th joint and for each class label.
            model.jointparts(i).sigma: as above

    """
    model=modelType()
    model.class_priors = np.zeros([4])
    model.jointparts = list()
    model.connectivity = G
    class_counter = 0
    #Compute class priors
    for cl in (1,2,3,8):
        prior = np.sum(labels[:,0]==cl) / len(labels)
        model.class_priors[class_counter]=prior
        class_counter=class_counter+1
    
    #Compute mean and variance 
    for ijoint in range (dataset.shape[0]):
        if G is None or G[ijoint]==-1:
            class_counter = 0
            jointp= jointPartsT()
            jointp.j_nb = NB_MODEL
            jointp.j_sigma = np.zeros([3,4])
            jointp.j_mean = np.zeros([3,4])
            jointp.j_betas = np.zeros([12,4])
            for cl in (1,2,3,8):
                class_selection = np.transpose(labels==cl)
                d_cl = dataset[:,:,class_selection[0]]
                
                jointp.j_mean[0,class_counter], jointp.j_sigma[0,class_counter] = fit_gaussian(d_cl[ijoint,0,:])
                jointp.j_mean[1,class_counter], jointp.j_sigma[1,class_counter] = fit_gaussian(d_cl[ijoint,1,:])
                jointp.j_mean[2,class_counter], jointp.j_sigma[2,class_counter] = fit_gaussian(d_cl[ijoint,2,:])
                
                class_counter = class_counter+1
                
            model.jointparts.append(jointp)
            
        else:
            class_counter = 0
            jointp= jointPartsT()
            jointp.j_nb = LG_MODEL
            jointp.j_sigma = np.zeros([3,4])
            jointp.j_mean = np.zeros([3,4])
            jointp.j_betas = np.zeros([12,4])
            for cl in (1,2,3,8):
                class_selection = np.transpose(labels==cl)
                d_cl = dataset[:,:,class_selection[0]]
                s, b = fit_linear_gaussian(d_cl[ijoint,0,:],np.transpose(d_cl[G[ijoint],:,:]))
                jointp.j_sigma[0,class_counter] = s[0]
                jointp.j_betas[0:4,class_counter:4]=b
                
                s, b = fit_linear_gaussian(d_cl[ijoint,1,:],np.transpose(d_cl[G[ijoint],:,:]))
                jointp.j_sigma[1,class_counter] = s[0]
                jointp.j_betas[4:8,class_counter:8]=b
                
                s, b = fit_linear_gaussian(d_cl[ijoint,2,:],np.transpose(d_cl[G[ijoint],:,:]))
                jointp.j_sigma[2,class_counter] = s[0]
                jointp.j_betas[8:12,class_counter:12]=b
                
                
                class_counter = class_counter+1
            model.jointparts.append(jointp)
            
    return model


def normalize_logprobs(log_probs):
    """
       Returns the log prob normalizes so that when exponenciated
       it adds up to 1 (Useful to normalizes logprobs)
    """
    mm = np.max(log_probs)
    return log_probs - mm - np.log(np.sum(np.exp(log_probs - mm)))

def log_normpdf(x, mu, sigma):
    """
      Computes the natural logarithm of the normal probability density function
      
    """
    diff = x-mu
    return np.log(1/(math.sqrt(2*math.pi*sigma)))-(math.pow(diff,2)/(2*sigma))

def compute_logprobs(example, model):
    """
       Input
           instance: a 20x3 matrix defining body positions of one instance
           model: as given by learn_model
       Output
           l: a vector of len #classes containing the loglikelihhod of the 
              instance
    """
    prob_per_class = np.zeros(NUMBER_OF_CLASSES)
    
    for cl in range (NUMBER_OF_CLASSES):
        prob_sum = 0
        for joint in range (example.shape[0]):
            
            for coord in range (example.shape[1]):
                sigma = model.jointparts[joint].j_sigma[coord,cl]
                if (model.jointparts[joint].j_nb == NB_MODEL):
                    mean = model.jointparts[joint].j_mean[coord,cl]
                else:
                    parent = model.connectivity[joint]
                    betas = model.jointparts[joint].j_betas[:,cl]
                    if coord == 0:                        
                        mean = betas[0]+betas[1]*example[parent,0]+betas[2]*example[parent,1]+betas[3]*example[parent,2]
                        
                    if coord == 1: 
                        mean = betas[4]+betas[5]*example[parent,0]+betas[6]*example[parent,1]+betas[7]*example[parent,2]
                        
                    if coord == 2: 
                        mean = betas[8]+betas[9]*example[parent,0]+betas[10]*example[parent,1]+betas[11]*example[parent,2]                
                    
                prob_sum = prob_sum+log_normpdf(example[joint,coord],mean,sigma)

        prob_per_class[cl] = prob_sum+np.log(model.class_priors[cl])
    
    
    norm_prob_per_class=normalize_logprobs(prob_per_class)
    test = np.exp(norm_prob_per_class)                    
                    
    return test

def classify_instances(instances, model):
    """    
    Input
       instance: a 20x3x#instances matrix defining body positions of
                 instances
       model: as the output of learn_model

    Output
       probs: a matrix of #instances x #classes with the probability of each
              instance of belonging to each of the classes

    Important: to avoid underflow numerical issues this computations should
               be performed in log space
    """
    num_instances = instances.shape[2]
    class_prob = np.zeros((num_instances,NUMBER_OF_CLASSES))
    for i in range (num_instances):
        class_prob[i,:]=compute_logprobs(instances[:,:,i],model)
        
    return class_prob


def translate_class_probs(data_classify):
    """
       Translate the raw probability of the data classified to a column with the 
       class value. In this case only 4 classes were considered
       Input
           data_classify: raw probabilities 
           
       Output
           res: column with class values
    """
    res = np.zeros([data_classify.shape[0],1])
    for i in range (data_classify.shape[0]):
        value = np.argmax(data_classify[i,:])
        if (value == 0):
            n_val = 1
        if (value == 1):
            n_val = 2
        if (value == 2):
            n_val = 3
        if (value == 3):
            n_val = 8
        res[i] = n_val
#    print(res.shape)
    return res


def get_model_accuracy(res_labels,real_labels):
    """
        Calculate the accuracy of the result: obtained againts expected labels
    """
    acc_counter = 0
    for index in range (res_labels.shape[0]):
        if (real_labels[index] == res_labels[index]):
            acc_counter = acc_counter+1
    accuracy = acc_counter/res_labels.shape[0]
    print(accuracy)
    return accuracy


def main():
    """
        Creates the model and evaluates it using K-fold cross validation
        Prints the accuracy of the model 
    """
    ######### Evaluate algorithm using cross validation
    
    #Select data to use 
    data,labels,individuals = load_dataset()
    #data = full_valid_set
    #labels = full_labels_set
    X = np.zeros(data.shape[2])
    num_splits = 10
    kf = KFold(n_splits=num_splits)
 
    #print cross validation configuration
    print(kf)
    cross_val = 0
    for train_index, test_index in kf.split(X):        
        data_train, data_test = data[:,:,train_index], data[:,:,test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        
        ####  Uncomment one of the following line deppending the type of connectivity to use
        #connection = None
        connection = nui_skeleton_conn
        ####

        #Create the model 
        model = learn_model(data_train,labels_train,connection)
        #Classify instances 
        data_classify = classify_instances(data_test,model)
        predicted_class = translate_class_probs(data_classify)
        accuracy = get_model_accuracy(predicted_class,labels_test)
        cross_val = cross_val + accuracy
        
    cross_val = cross_val/num_splits
    print("final accuracy estimated with cross-valid (k_fold) = ",cross_val)
        


if __name__ == '__main__':
    main()