# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:19:30 2019

@author: Santi
"""

import scipy.io
import numpy as np
import statistics as st
from dataclasses import dataclass

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


def fit_gaussian(v_observed):
    "Compute mean and stdrd deviation of a vector of observations for a variable"
    mean_value = np.mean(v_observed)
    std_value = np.std(v_observed)
    return mean_value, std_value

def fit_linear_gaussian(Y, X, W):
    "Obtain the vector of betas as long as the input vector and input matrix have the same number of rows"
    if len(Y) != len(X):
        print("ERROR!!! The length of the vectors doesn't match")
    "To create a row of ones at the left of the matrix"
    X_new = np.insert(X,0,1,axis=1)
    "For creating A, we multiply the new X by its transpose"
    X_t = np.transpose(X_new)
    "Creation of a square matrix with as many rows and columns as variables there were plus one"
    A = np.zeros((X_new.shape[1],X_new.shape[1]))
    "Creating a vector that contains as many variables as we had plus one"
    b = np.zeros((X_new.shape[1],1))
    "For creating b, we transpose the vector Y"
    Y_t = np.transpose(Y)
    "Filling of the A matrix"
    A = np.matmul(X_t,X_new)
    "We have to normalize the matrix"
    A = A/X.shape[0]
    "Calculating the b vector"
    b = np.matmul(Y_t,X_new)
    "We have to normalize the vector"
    b = b/X.shape[0]    
    "Solve the betas vector"
    betas_vec = np.linalg.solve(A,np.transpose(b))
    "Calculate the covariances"
    counter = 0
    "Calculate the covariance of Y"
    covY = my_cov(Y,Y,W)
    "Calculate the covariances of X and Y conditioned by betas"
    for i in range(1, betas_vec.shape[0]):
        for j in range(1, betas_vec.shape[0]):
            c = betas_vec[i]*betas_vec[j]*(np.mean((X[:,i-1] - np.mean(X[:,i-1]))*((X[:,j-1] - np.mean(X[:,j-1])))))
            counter +=c
    "Standard deviation relating the covariance of Y and the sum of X-Y covariances"
    sigmasquare = np.sqrt(covY - counter)
    return betas_vec, sigmasquare

def my_cov(x,y,w):
    return np.sum(w*x*y)/np.sum(w)-np.sum(w*x)*np.sum(w*y)/np.sum(w)/np.sum(w)

@dataclass
class Model:
    connectivity: list()
    priors: np.array
    jointparts: list()
    def __init__(self, params=None):
        pass

@dataclass
class JointParts:
    j_mean: np.array
    j_sigma: np.array
    j_betas: np.array
    
    def __init__(self, params=None):
        pass 
    
def learn_model(dataset, labels,G=None):
    model=Model()
    model.class_priors = np.zeros([4])
    model.jointparts = list()
    model.connectivity = G
    class_counter = 0
    #Compute class priors
    for cl in (1,2,3,8):
        prior = np.sum(labels[:,0]==cl) / len(labels)
        print(prior)
        model.class_priors[class_counter]=prior
        class_counter=class_counter+1
    
    #Compute mean and variance 
    for ijoint in range (dataset.shape[0]):
        if G is None or G[ijoint]==-1:
            class_counter = 0
            jointp= JointParts()
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
            jointp= JointParts()
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
                jointp.j_betas[8:12,class_counter:12]=b #Por que el segundo param hasta 12?
                
                
                class_counter = class_counter+1
            model.jointparts.append(jointp)
            
    return model


def normalize_logprobs(log_probs):
    "Returns the log prob normalizes so that when exponenciated it adds up to 1 (Useful to normalizes logprobs)",
    mm = np.max(log_probs)
    return log_probs - mm - np.log(np.sum(np.exp(log_probs - mm)))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"Aqui empieza la magia santiniana"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def log_normpdf(x, mu, sigma):
    return #natural logarithm of the normal probability density function

def compute_logprobs(example, model):
    log_probs = np.zeros((1,4))
    class_counter = 0
    for cl in (1,2,3,8):
        #Compute the probability of the instance to belong to each class
        #No seguro de si el primer elemento es las betas
        log_probs[1,cl] = log_normpdf(example[i][j], model.jointparts.j_betas[:,class_counter], model.jointparts.j_mean[:,class_counter], model.jointparts.j_sigma[:,class_counter])
        class_counter = class_counter+1
    return log_probs

def classify_instances(instances, model):
    log_vector = np.zeros((instances.shape[0],4))
    for inst in range (0, instances.shape[0]-1):
        log_vector[inst,:] = compute_logprobs(instances[inst], model)
    postprob = normalize_logprobs(log_vector)
    return postprob

def main():
    #data = np.array(sc.loader(os.path.join(execution_path, "template_data\data\data.mat")))
    
dd = scipy.io.loadmat('./template_data/data/validation_data.mat')
dataset = dd['data_small'] # Input data
labels = dd['labels_small'] # Input labels
dd['individuals_small'] # Input individual indexes
dd['train_indexes'] # Instances used for training
dd['test_indexes']  # Instances used for test
dd['model_nb']      # NB model
dd['model_lg']      # LG model
dd['accur_nb']      # Accuracy of NB model on test instances
dd['accur_lg']      # Accuracy of LG model on test instances
  
learn_model(dataset, labels, G)
        
if __name__ == '__main__':
    main()
