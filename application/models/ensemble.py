import numpy as np

class DTNode:
    """ Decision Tree Node """
    N_THRESHOLD = 4                 # don't split if node has fewer examples than this
    H_THRESHOLD = .01               # don't split if node has entropy less than this
    H_REDUCTION_THRESHOLD = .001    # don't split if it doesn't reduce H by this
    MAX_DEPTH = 10                  # maximum depth of the tree

    index = 0
    tree_depth = 0                  # current depth of the whole tree

    def __init__(self, data=None, config=None, depth=0):
        """ Create a node in the decision tree. """
        
        self.config = config
        if config != None:
            self.N_THRESHOLD = config[0]
            self.H_THRESHOLD = config[1]
            self.H_REDUCTION_THRESHOLD = config[2]
            self.MAX_DEPTH = config[3]

        DTNode.index += 1
        self.index = DTNode.index           # unique number for each node
        self.data = data                    # data at the node
        self.prob = None                    # probability of positive label (label = 1) for the data at the node
        self.depth = depth                  # depth of the node in the tree
        if self.depth > DTNode.tree_depth:
            DTNode.tree_depth = self.depth

        if data is not None:
            self.n = float(data.shape[0])          # number of data points at the node
            self.indices = range(data.shape[1]-1)  # indices of features (used for splitting)
            self.set_h()                           # compute entropy of the node

        self.splits = {}    # splits for each feature (key = feature index, value = C{List} of thresholds)
        self.fi = None      # feature index of the best split
        self.th = None      # threshold of the best split
        self.lc = None      # left child node
        self.rc = None      # right child node
        self.parent = None  # parent node

    def set_h(self):
        """ Set the entropy of the node, assumes labels are 1, -1 """

        b = .001
        npos = np.sum(self.data[:, -1] == 1)  # count labels = 1
        prob = (npos + b) / (self.n + b + b)
        self.prob = prob
        self.H = -prob*np.log(prob) - (1-prob)*np.log(1-prob)

    def build_tree(self):
        """ Build the tree recursively """

        # don't split if entropy is low, or if there are few examples, or if we're too deep
        if self.H < self.H_THRESHOLD or self.n < self.N_THRESHOLD or self.depth >= self.MAX_DEPTH:
            return

        # find best split (with max information gain)
        (i, th, (h, lc, rc)) = self.argmax([(i, th, self.split_eval(i, th)) \
                                                for i in self.indices       \
                                                for th in self.get_splits(i)], 
                                           lambda x: self.H - x[2][0])

        # don't split if it doesn't reduce entropy enough
        if self.H - h < self.H_REDUCTION_THRESHOLD:
            return
        
        self.fi = i
        self.th = th
        self.lc = lc
        self.rc = rc
        self.lc.parent = self
        self.rc.parent = self
        
        # recursively build children
        self.lc.build_tree()
        self.rc.build_tree()

    def get_splits(self, i):
        """ Find the best splitting point for data at node along feature at index i """

        if i not in self.splits:
            self.splits[i] = np.sort(np.unique(self.data[:,i]), axis=None)
        return self.splits[i]

    def split_eval(self, i, th):
        """ Evaluate weighted average entropy of splitting feature at index i by threshold th """

        lc = DTNode(self.data[self.data[:, i] < th], config=self.config, depth=self.depth+1)
        rc = DTNode(self.data[self.data[:, i] >= th], config=self.config, depth=self.depth+1)
        pl = lc.n / self.n
        pr = 1.0 - pl
        avgH = pl*lc.H + pr*rc.H

        return avgH, lc, rc
    
    def classify(self, x):
        """ Classify a single example """

        # return probability of positive label if leaf node
        if self.fi == None:
            return self.prob            
        if x[self.fi] < self.th:
            return self.lc.classify(x)
        else:
            return self.rc.classify(x)

    def display(self, depth=0, max_display_depth=3):
        """ Display the tree """

        if depth > max_display_depth:
            print(depth*'  ', 'Depth >', max_display_depth)
            return
        if self.fi is None:
            print('\033[32m', end="")
            print(depth*'  ', f'[L={depth}]', "%.2f"%self.prob, '[', 'n=', "%.0f"%self.n, ']')
            print('\033[0m', end="")            
            return
        print(depth*'  ', f'[N={depth}]', 'fi', self.fi, '< th=', "%.5f"%self.th, '[', 'n=', "%.0f"%self.n, ']')
        self.lc.display(depth+1, max_display_depth)
        self.rc.display(depth+1, max_display_depth)

    @staticmethod
    def argmax(l, f):
        """
        @param l: C{List} of items
        @param f: C{Procedure} that maps an item into a numeric score
        @returns: the element of C{l} that has the highest score
        """
        vals = [f(x) for x in l]
        return l[vals.index(max(vals))]

class DecisionTree:
    """ Decision Tree Classifier """

    def fit(self, X, Y, config=None):
        D = np.hstack([X,Y])
        self.root = DTNode(D, config=config)
        self.root.build_tree()
        return self.root
        
    def predict(self, X):
        pred = np.array([np.apply_along_axis(self.root.classify, 1, X)]).T - 0.5
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred
    
    def display(self, depth=0, max_depth=3):
        self.root.display(depth, max_depth)

class Bagging:
    """ Bagging (Booststrap Aggregation) Classifier """
    
    def __init__(self, num_trees=5):
        self.ntrees = num_trees
        self.trees = []
        
    def fit(self, X, Y, config=None):
        for i in range(self.ntrees):
            idxs = np.random.choice(len(X), size=len(X), replace=True)
            X_train = X[idxs, :]
            Y_train = Y[idxs, :]
            dt = DecisionTree()
            dt.fit(X_train, Y_train, config)
            self.trees.append(dt)
            
    def predict(self, X):
        preds = []
        if len(self.trees) == 0: return None
        for dt in self.trees:
            pred = dt.predict(X)
            preds.append(pred)
        preds = np.hstack(preds)
        return np.sign(preds.mean(axis=1, keepdims=True))

class RandomForest:
    """ Random Forest Classifier """
    
    def __init__(self, num_trees=5, num_features=None):
        self.ntrees = num_trees
        self.nfeats = num_features
        self.trees = []
        self.feats = []
    
    def fit(self, X, Y, config=None):
        for i in range(self.ntrees):
            idxs = np.random.choice(len(X), size=len(X), replace=True)
            if self.nfeats is not None:
                features = np.random.choice(X.shape[1], size=self.nfeats, replace=False)
                X_train = X[idxs][:, features]
                self.feats.append(features)
            else:
                X_train = X[idxs]
            Y_train = Y[idxs]
            dt = DecisionTree()
            dt.fit(X_train, Y_train, config)
            self.trees.append(dt)
    
    def predict(self, X):
        preds = []
        if len(self.trees) == 0:
            return None
        for i in range(len(self.trees)):
            dt = self.trees[i]
            if self.nfeats is not None:
                features = self.feats[i]
                X_test = X[:, features]
            else:
                X_test = X
                
            pred = dt.predict(X_test)
            preds.append(pred)
        preds = np.hstack(preds)
        return np.sign(np.mean(preds, axis=1, keepdims=True))
