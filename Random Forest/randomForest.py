
#---------------Instructions------------------#
# Please read the function documentation before
# proceeding with code writing. 

# For randomizing, you will need to use following functions
# please refer to their documentation for further help.
# 1. np.random.randint
# 2. np.random.random
# 3. np.random.shuffle
# 4. np.random.normal 


# Other Helpful functions: np.atleast_2d, np.squeeze()
# scipy.stats.mode, np.newaxis

#-----------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#
#%pylab inline
'''
import scipy.stats
import matplotlib.pyplot as plt
from collections import defaultdict  # default dictionary 
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
#%load_ext autoreload
'''

import tree as tree
import numpy as np
#import pandas as pd
#import tools as t
#import time



class RandomForest:
    ''' Implements the Random Forest For Classification... '''
    def __init__(self, ntrees=10,treedepth=5,usebagging=False,baggingfraction=0.6,
        weaklearner="Conic",
        nsplits=10,        
        nfeattest=None, posteriorprob=False,scalefeat=True ):        
        """      
            Build a random forest classification forest....

            Input:
            ---------------
                ntrees: number of trees in random forest
                treedepth: depth of each tree 
                usebagging: to use bagging for training multiple trees
                baggingfraction: what fraction of training set to use for building each tree,
                weaklearner: which weaklearner to use at each interal node, e.g. "Conic, Linear, Axis-Aligned, Axis-Aligned-Random",
                nsplits: number of splits to test during each feature selection round for finding best IG,                
                nfeattest: number of features to test for random Axis-Aligned weaklearner
                posteriorprob: return the posteriorprob class prob 
                scalefeat: wheter to scale features or not...
        """

        self.ntrees=ntrees
        self.treedepth=treedepth
        self.usebagging=usebagging
        self.baggingfraction=baggingfraction

        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.nfeattest=nfeattest
        
        self.posteriorprob=posteriorprob
        
        self.scalefeat=scalefeat
        
        pass

    def findScalingParameters(self,X):
        """
            find the scaling parameters
            input:
            -----------------
                X= m x d training data matrix...
        """
        self.mean=np.mean(X,axis=0)
        self.std=np.std(X,axis=0)

    def applyScaling(self,X):
        """
            Apply the scaling on the given training parameters
            Input:
            -----------------
                X: m x d training data matrix...
            Returns:
            -----------------
                X: scaled version of X
        """
        X= X - self.mean
        X= X /self.std
        return X

    def train(self,X,Y,vX=None,vY=None):
        '''
            Trains a RandomForest using the provided training set..
        
            Input:
            ---------
            X: a m x d matrix of training data...
            Y: labels (m x 1) label matrix

            vX: a n x d matrix of validation data (will be used to stop growing the RF)...
            vY: labels (n x 1) label matrix

            Returns:
            -----------

        '''

        nexamples, nfeatures= X.shape

        self.findScalingParameters(X)
        if self.scalefeat:
            X=self.applyScaling(X)

        self.trees=[]
            
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        self.classes = np.unique(Y)
        for i in range(self.ntrees):
            sort = np.arange(0,nexamples)
            mixed_val = np.random.shuffle(sort)
            mixed_valX = np.squeeze(X[mixed_val])
            mixed_valY = np.squeeze(Y[mixed_val])
            
            dt = tree.DecisionTree(purity=0.9,maxdepth=self.treedepth,weaklearner=self.weaklearner,nsplits=self.nsplits,nfeattest=nfeatures)
            dt.train(mixed_valX,mixed_valY)
            self.trees.append(dt)
        
        #---------End of Your Code-------------------------#
        
    def predict(self, X):
        
        """
        Test the trained RF on the given set of examples X
        
                   
            Input:
            ------
                X: [m x d] a d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z = []
        
        if self.scalefeat:
            X=self.applyScaling(X)

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        nexamples,nfeatures = X.shape
        tree_pred = []
        for intree in self.trees:
            tree_pred.append(intree.predict(X))
        
        pred = np.array(tree_pred)
        print(pred)
        for i in range(0, nexamples):
            min_val = -np.inf
            feat_idx = -1
            for j in range(0, len(self.classes)):
                flag = (pred[:,i] == self.classes[j])
                flag_sum = np.sum(flag)
                if (flag_sum > min_val):
                    feat_idx = j
                    min_val = flag_sum
            z.append(self.classes[feat_idx])
       
        return z 
        
        #---------End of Your Code-------------------------#

''''
Used for testing
#generate training and testing set...by sampling from mutli-variate Gaussian
np.random.seed(seed=99)
cp=5
nclasses=2
mean1 = [-cp,-cp]
mean2 = [cp,cp]
mean3 = [cp,-cp]
mean4 = [-cp,cp]
cov = [[3.0,0.0],[0.0,3.0]] 

#create some points for the training set...
nexamples=2000
size=int(nexamples/4)
x1 = np.random.multivariate_normal(mean1,cov,[size])
x2 = np.random.multivariate_normal(mean2,cov,[size])
x3 = np.random.multivariate_normal(mean3,cov,[size])
x4 = np.random.multivariate_normal(mean4,cov,[size])

X=np.vstack((x1,x2,x3,x4))
Y=np.vstack((1*np.ones((size,1)),2*np.ones((size,1)),3*np.ones((size,1)),4*np.ones((size,1))))


plt.scatter(x1[:,0],x1[:,1], c='r', s=100)
plt.scatter(x2[:,0],x2[:,1], c='g', s=100)            
plt.scatter(x3[:,0],x3[:,1], c='b', s=100)
plt.scatter(x4[:,0],x4[:,1], c='y', s=100)            



plt.title("Multi-class Classification")
plt.xlabel("feature $x_1$")
plt.ylabel("feature $x_2$")
plt.legend(['r','g','b','y'])
fig_ml_in_10 = plt.gcf()
plt.savefig('multi-linear-class.svg',format='svg')
#create some points for the training set..

ntexamples=1000
size=int(ntexamples/4)
x1 = np.random.multivariate_normal(mean1,cov,[size])
x2 = np.random.multivariate_normal(mean2,cov,[size])
x3 = np.random.multivariate_normal(mean3,cov,[size])
x4 = np.random.multivariate_normal(mean4,cov,[size])

Xt=np.vstack((x1,x2,x3,x4))
Yt=np.vstack((1*np.ones((size,1)),2*np.ones((size,1)),3*np.ones((size,1)),4*np.ones((size,1))))

rfc = RandomForest()
print (X.shape, Y.shape)
print(np.unique(X))
rfc.train(X,Y)
print(len(rfc.trees))

Yp=rfc.predict(X)
#print (Y.shape, len(Yp))
#t.print_confusion_matrix(Yp,Y)

#acc = np.sum(Y ==  np.array(Yp)) / float(Y.shape[0])
#print (acc)
'''