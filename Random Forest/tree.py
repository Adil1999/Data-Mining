
# A good heuristic is to choose sqrt(nfeatures) to consider for each node...
import weakLearner as wl
import numpy as np
#import pandas as pd

#---------------Instructions------------------#

# Here you will have to reproduce the code you have already written in
# your previous assignment.

# However one major difference is that now each node non-terminal node of the
# tree  object will have  an instance of weaklearner...

# Look for the missing code sections and fill them.
#-------------------------------------------#

class Node:
    def __init__(self,purity,klasslabel='',pdistribution=[],score=0,wlearner=None): # purity = score and score = split value
        """
               Input:
               --------------------------
               klasslabel: to use for leaf node
               pdistribution: posteriorprob class probability at the node
               score: split score
               weaklearner: which weaklearner to use this node, an object of WeakLearner class or its childs...

        """

        self.lchild=None
        self.rchild=None
        self.klasslabel=klasslabel
        self.pdistribution=pdistribution
        self.score=score
        self.wlearner=wlearner
        self.purity = purity

    def set_childs(self,lchild,rchild):
        
        self.lchild=lchild
        self.rchild=rchild

        
    def isleaf(self):
        # YOUR CODE HERE
        if (self.rchild==None and self.lchild==None):
            return True
        else:
            return False

    def isless_than_eq(self, X):
       
        return self.wlearner.evaluate(X)
        #---------End of Your Code-------------------------#

    def get_str(self):
        """
            returns a string representing the node information...
        """
        if self.isleaf():
            return 'C(posterior={},class={},Purity={})'.format(self.pdistribution, self.klasslabel,self.purity)
        else:
            return 'I(Fidx={},Score={},Split={})'.format(self.fidx,self.score,self.split)


class DecisionTree:
    ''' Implements the Decision Tree For Classification With Information Gain
        as Splitting Criterion....
    '''
    def __init__(self, purity = 0.8, exthreshold=5, maxdepth=10,
     weaklearner="Conic", pdist=False, nsplits=10, nfeattest=None):
        '''
        Input:
        -----------------
            exthreshold: Number of examples to stop splitting, i.e. stop if number examples at a given node are less than exthreshold
            maxdepth: maximum depth of tree upto which we should grow the tree. Remember a tree with depth=10
            has 2^10=1K child nodes.
            weaklearner: weaklearner to use at each internal node.
            pdist: return posterior class distribution or not...
            nsplits: number of splits to use for weaklearner
        '''
        self.purity = purity
        self.maxdepth=maxdepth
        self.exthreshold=exthreshold
        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.pdist=pdist
        self.nfeattest=nfeattest
        assert (weaklearner in ["Conic", "Linear","Axis-Aligned","Axis-Aligned-Random"])
        #pass

    def getWeakLearner(self):
        if self.weaklearner == "Conic":
            return wl.ConicWeakLearner(self.nsplits)
        elif self.weaklearner== "Linear":
            return wl.LinearWeakLearner(self.nsplits)
        elif self.weaklearner == "Axis-Aligned":
            return wl.WeakLearner()
        else:
            return wl.RandomWeakLearner(self.nsplits,self.nfeattest)

        #pass

    def train(self, X, Y):
        ''' Train Decision Tree using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns:
            -----------
            Nothing
            '''
        nexamples,nfeatures=X.shape
        ## now go and train a model for each class...
        # YOUR CODE HERE
        
        self.tree = self.build_tree(X,Y,self.maxdepth)   
        

    def build_tree(self, X, Y, depth):
        """ 
            Function is used to recursively build the decision Tree 
          
            Input
            -----
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns
            -------
            root node of the built tree...
        """
        nexamples, nfeatures=X.shape
        # YOUR CODE HERE      
        
        classes, class_counts = np.unique(Y, return_counts = True);
        prob_dist = []
        for i in range(len(class_counts)):
            prob_dist.append(class_counts[i]/float(nexamples))
        klass = None
        root = None
        
        if(nexamples <= self.exthreshold or depth <= 1):
            index = np.argmax(prob_dist)
            klass = classes[index]
            root = Node(prob_dist[index],klass,prob_dist)
            return root
        else:
            weak_learner = self.getWeakLearner()
            purity, score, child_L, child_R = weak_learner.train(X,Y)
            root = Node(purity, klass, prob_dist, score, weak_learner)    
            root = self.grow_tree(root, X, Y, child_L, child_R, depth)
        return root      
    
    def grow_tree(self,root, X, Y, child_L, child_R, depth):
        left_size = len(child_L)
        right_size = len(child_R)
        
        if (left_size > 1 and right_size > 1):
            grow_to_right = self.build_tree(X[child_R,:],Y[child_R], depth-1)
            grow_to_left = self.build_tree(X[child_L,:],Y[child_L], depth-1)
            root.set_childs(grow_to_left, grow_to_right)
            
        elif left_size > 1:
            grow_to_left = self.build_tree(X[child_L,:],Y[child_L],depth-1)
            root.set_childs(grow_to_left,None)
        
        elif right_size > 1:
            grow_to_right = self.build_tree(X[child_R,:],Y[child_R],depth-1)
            root.set_childs(None,grow_to_right)
        
        return root    
        
    def test(self, X):
        
        ''' Test the trained classifiers on the given set of examples 
        
                   
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for each example, i.e. to which it belongs
        '''
        
        nexamples, nfeatures = X.shape
        pclasses = self.predict(X)
        return np.array(pclasses)

    def predict(self, X):

        """
        Test the trained classifiers on the given example X


            Input:
            ------
            X: [1 x d] a d-dimensional test example.

            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        pred = []
        
        for index in range(X.shape[0]):
            pred.append(self._predict(self.tree, np.atleast_2d(X[index,:])))
        
        return pred

    def _predict(self,node, X):
        # YOUR CODE HERE
        
        if node.isleaf():
            return node.klasslabel
        else:
            if node.lchild != None and node.isless_than_eq(X):
                return self._predict(node.lchild,X)
            elif node.rchild != None:
                return self._predict(node.rchild,X) 
        #---------End of Your Code-------------------------#


    def __str__(self):
        """
            overloaded function used by print function for printing the current tree in a
            string format
        """
        str = '---------------------------------------------------'
        str += '\n A Decision Tree With Depth={}'.format(self.find_depth())
        str += self.__print(self.tree)
        str += '\n---------------------------------------------------'
        return str  # self.__print(self.tree)


    def _print(self, node):
        """
                Recursive function traverse each node and extract each node information
                in a string and finally returns a single string for complete tree for printing purposes
        """
        if not node:
            return
        if node.isleaf():
            return node.get_str()

        string = node.get_str() + self._print(node.lchild)
        return string + node.get_str() + self._print(node.rchild)

    def find_depth(self):
        """
            returns the depth of the tree...
        """
        return self._find_depth(self.tree)

    def _find_depth(self, node):
        """
            recursively traverse the tree to the depth of the tree and return the depth...
        """
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild), self._find_depth(node.rchild)) + 1

    def __print(self, node, depth=0):
        """

        """
        ret = ""

        # Print right branch
        if node.rchild:
            ret += self.__print(node.rchild, depth + 1)

        # Print own value

        ret += "\n" + ("    "*depth) + node.get_str()

        # Print left branch
        if node.lchild:
            ret += self.__print(node.lchild, depth + 1)

        return ret
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

dt = DecisionTree()
dt.train(X,Y)
'''