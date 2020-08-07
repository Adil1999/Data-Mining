

#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes.
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
#import pandas as pd

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...

    
    """
    
    def __init__(self):
        self.split = 0
        self.fidx = -1

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures = X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        Xlidx = []
        Xridx = []
        score = 999999
        split = -999999
        fidx = -1
        
        for i in range(0,nfeatures):
            new_split, new_score, l_idx, r_idx = self.evaluate_numerical_attribute(X[:,i],Y)
            if(new_score < score):
                score = new_score
                split = new_split
                fidx = i
                Xlidx = l_idx
                Xridx = r_idx
            #print(r_idx)
        self.split = split
        self.fidx = fidx
        
        #---------End of Your Code-------------------------#
        
        return split, score, Xlidx, Xridx

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        if X.shape[0] == 1:
            X = X.flatten()
        
        if(X[self.fidx] < self.split):
            return True
        else:
            return False

        #---------End of Your Code-------------------------#
        
    def find_column_index(self, sY, itr, classes, nclasses):
        class_idx = 0
        for j in range(nclasses):    #finding index of class in which it lies
            if(sY[itr] == classes[j]):
                class_idx = j
                break
        return class_idx, j
    
    def calculate_entropies(self, itr, classCounts, no_of_splits):
        denom1 = np.sum(no_of_splits[itr,:])
        denom2 = np.sum(classCounts) - denom1
        logSum = [0,0]
        temp = 0
        for j in range(0, no_of_splits.shape[1]):
            temp = classCounts[j] - no_of_splits[itr][j]
            if(no_of_splits[itr][j] != 0):
                logSum[0] = logSum[0] + ((no_of_splits[itr][j] / denom1) * (np.log2(no_of_splits[itr][j] / denom1)))
            if(temp != 0):
                logSum[1] = logSum[1] + ((temp/denom2) * (np.log2(temp/denom2)))
        temp = denom1 + denom2
        logSum[0] = -1 * logSum[0]
        logSum[1] = -1 * logSum[1]
        
        return logSum[0], logSum[1], temp, denom1, denom2
    
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        
        classes, classCounts = np.unique(Y, return_counts = True) # unique_classes and their counts
        self.classes = classes
        nclasses = len(classes)
        sidx = np.argsort(feat)
        f = feat[sidx] # sorted features
        sY = Y[sidx] # sorted features class labels...
        f_size = len(f)
        # YOUR CODE HERE
        
        #calculating midpoints
        mid_points = np.zeros(f_size)
        for i in range(1, f_size): mid_points[i] = (f[i]+f[i - 1]) / 2
        mid_points = np.unique(mid_points)
        midpoints_size = len(mid_points)
        #print(mid_points)
        
        #calculating number of splits
        no_of_splits = np.zeros((midpoints_size,nclasses))
        for index in range(midpoints_size):
            for i in range(f_size):
                class_idx, j = self.find_column_index(sY, i, classes, nclasses)
                if(f[j] < mid_points[index]): no_of_splits[index][class_idx] += 1
        #print(no_of_splits)        
        
        #calculating entropies
        entropies = np.zeros(midpoints_size)
        for i in range(0, no_of_splits.shape[0]):
            log_val1, log_val2, aggregate,denom1,denom2 = self.calculate_entropies(i, classCounts, no_of_splits)
            entropies[i] = ((denom1/float(aggregate)) * log_val1) + ((denom2/float(aggregate)) * log_val2)                
        #print(entropies)
        
        entropies = entropies[entropies > 0]
        min_entropy = np.argmin(entropies)
        Xlidx = []
        Xridx = []    
        for dp in range(f_size):
            if(f[dp] < mid_points[min_entropy]):
                Xlidx.append(dp)
            else:
                Xridx.append(dp)  
        
        #---------End of Your Code-------------------------#
        split = mid_points[min_entropy] 
        score = entropies[min_entropy]
        Xlidx = np.array(Xlidx)
        Xridx = np.array(Xridx)
        
        return split, score, Xlidx, Xridx

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using
        a random set of features from the given set of features...

    """

    def __init__(self, nsplits = +np.inf, nrandfeat = None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...
        self.nsplits = nsplits
        self.nrandfeat = nrandfeat
        self.fidx = -1
        self.split = -1
        #pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        #print "Inside the train of Random"
        nexamples,nfeatures = X.shape

        #print "Train has X of length ", X.shape


        if(not self.nrandfeat):
            self.nrandfeat = int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        score = 99999
        split = -99999
        fidx = -1
        child_L = []
        child_R = []
        idx = np.random.randint(0, nfeatures,self.nrandfeat)
        for i in range(0, self.nrandfeat):
            if(self.nsplits != 99999):
                new_split, new_score, Xlidx, Xridx = self.findBestRandomSplit(X[:,idx[i]],Y)
            else:
                new_split, new_score, Xlidx, Xridx = self.evaluate_numerical_attribute(X[:,idx[i]],Y)
            if(new_score < score):
                score = new_score
                child_L = Xlidx
                child_R = Xridx
                split = new_split
                fidx= idx[i]
            self.split = split
            self.fidx = fidx


        #---------End of Your Code-------------------------#
        return split, score, child_L, child_R

    def findBestRandomSplit(self,feat,Y):
        """

            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange = np.max(feat)-np.min(feat)


        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        #print(feat)
        random_splits = ((np.random.rand(1,feat.shape[0])) * frange)
        random_splits = random_splits.flatten()
        score = 99999
        best_split = -99999
        for split in random_splits:
            print(split)
            mship = feat < split
            temp = self.calculateEntropy(Y,mship)
            if(temp < score):
                score = temp
                best_split = split
        
        Xlidx = []
        Xridx = []
        for index in range(len(feat)):
            if(feat[index] >= best_split):
                Xridx.append(index)
            else:
                Xlidx.append(index)
        #---------End of Your Code-------------------------#
        
        Xlidx = np.array(Xlidx)
        Xridx = np.array(Xridx)
        
        print("rndm",best_split)
        
        return best_split, score, Xlidx, Xridx

    def get_entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)

        probabilities = counts / counts.sum() #Calculation probabilities of data occuring by it's sample space
        entropy = sum(probabilities * -np.log2(probabilities)) #using entropy formula
        
        return entropy
    
    def calculateEntropy(self, Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """
        child_L = Y[mship]
        child_R = Y[np.logical_not(mship)]
        
        #print(child_L)
        #print(child_R)
        
        prob_L = len(child_L) / float(len(Y))
        prob_R = 1 - prob_L
        
        sentropy =  (prob_L * self.get_entropy(child_L) 
                      + prob_R * self.get_entropy(child_R))
        return sentropy



    def evaluate(self, X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
                            
        if X.shape[0] == 1:
            X = X.flatten()
        
        if(X[self.fidx] < self.split):
            return True
        else:
            return False

        #---------End of Your Code-------------------------#

# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        self.a=0
        self.b=0
        self.c=0
        self.F1=0
        self.F2=0
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...

        """
        RandomWeakLearner.__init__(self, nsplits)
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible

            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape
        
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
    
        score = +99999
        split = -99999
        fidx = -1
        for i in range(self.nsplits):
            features = np.random.randint(0, nfeatures,2)
            a = np.random.normal()
            b = np.random.normal()
            c = np.random.normal()
            result = (X[:,features[0]] * a) + (X[:,features[1]] * b) + c
            mship = (result <= 0)
            '''
            if(self.nsplits != 99999):
                new_split, new_score, Xlidx, Xridx = self.findBestRandomSplit(X[:,idx[i]],Y)
            else:
                new_split, new_score, Xlidx, Xridx = self.evaluate_numerical_attribute(X[:,idx[i]],Y)
            '''
            new_score = self.calculateEntropy(Y,mship)
            if(new_score < score):
                fidx = features
                self.a = a
                self.b = b
                self.c = c
                score = new_score
                split = result
        
        #bXlm = []
        #for i in range(len(split)):
        #    if split[i] <= 0:
        #        bXlm.append(i)
        #print(bXlm)
        
        bXl = np.argwhere(split <= 0)
        #print(bXl)
        bXr = np.argwhere(split > 0)
        bXl = bXl.flatten()
        bXr = bXr.flatten()
        self.F1 = fidx[0]
        self.F2 = fidx[1]
                            
        #---------End of Your Code-------------------------#
        
        return 0, score, bXl, bXr  


    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        if X.ndim != 1:
            result = (X[:,self.F1] * self.a) + (X[:,self.F2] * self.b) + self.c
        elif X.ndim == 1:
            result = (X[:,self.F1] * self.a) + (1 * self.b) + self.c
            
        if(result > 0):
            return False
        else:
            return True

        #---------End of Your Code-------------------------#


#build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        self.a=0
        self.b=0
        self.c=0
        self.d=0
        self.e=0
        self.f=0
        self.F1=0
        self.F2=0
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #a, b, c, d, e, f = np.random.uniform(-3, 3, (6,)) going with my way 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        score = 99999
        split = -99999
        fidx = -1
        best_split_bounds = []
        #print(nsplits)
        for i in range (0, self.nsplits):
            features = np.random.randint(0,nfeatures,2)
            a = np.random.normal()
            b = np.random.normal()
            c = np.random.normal()
            d = np.random.normal()
            e = np.random.normal()
            f = np.random.normal()
            #classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+
            result = ((X[:,features[0]] ** 2) * a) + ((X[:,features[1]] ** 2) * b) + ((X[:,features[0]] * X[:,features[1]]) * c) + (X[:,features[0]] * d) + (X[:,features[1]] * e) + f
            split_bounds = np.random.normal(size=(2,1))
            if(np.random.random(1) < 0.5):
                split_bounds[0] = -99999
            mship = np.logical_and(result >= split_bounds[0], result < split_bounds[1])
            newScore = self.calculateEntropy(Y,mship)
            '''
            if(self.nsplits != 99999):
                new_split, new_score, Xlidx, Xridx = self.findBestRandomSplit(X[:,idx[i]],Y)
            else:
                new_split, new_score, Xlidx, Xridx = self.evaluate_numerical_attribute(X[:,idx[i]],Y)
            '''
            if(newScore < score):
                fidx = features
                self.a = a
                self.b = b
                self.c = c 
                self.d = d
                self.e = e
                self.f = f
                score = newScore
                split = result
                #print("conix",split.shape)
                best_split_bounds = split_bounds
        
        self.F1 = fidx[0]
        self.F2 = fidx[1]
        self.split = best_split_bounds
        self.score = score
        
        
        bXl = np.argwhere(split <= 0)
        bXr = np.argwhere(split > 0)
        bXl = bXl.flatten()
        bXr = bXr.flatten()
        #---------End of Your Code-------------------------#
        return 0, score, bXl, bXr 

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
                            
        X=X.flatten()
        result = ((X[self.F1] ** 2) * self.a) + ((X[self.F2] ** 2) * self.b) + ((X[self.F1] * X[self.F2]) * self.c) + (X[self.F1] * self.d) + (X[self.F2] * self.e) + self.f
                            
        if(result >= self.split[0] and result < self.split[1]):
            return True
        else:
            return False
        #---------End of Your Code-------------------------#
        
    
'''   
#load the data set
data = pd.read_csv('../iris.data')
data.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Class']
# Get your data in matrix (X ,Y)
Y = data.iloc[:,-1].values
#print(Y)
X = data.iloc[:, :4].values
#print(X)    
print (" Data Set Dimensions=", X.shape, " True Class labels dimensions", Y.shape) 


wki = WeakLearner()
#wki.train(X,Y)
wk = RandomWeakLearner(wki)
#wk.train(X, Y)
linear = LinearWeakLearner(5)
linear.train(X,Y)
#conic = ConicWeakLearner(5)
#conic.train(X,Y)
'''
