import numpy as np

# Decision tree node class
class DTNode:
    N_THRESHOLD = 4 # don't split if node has fewer examples than this
    H_THRESHOLD = .01 # don't split if node has entropy less than this
    H_REDUCTION_THRESHOLD = .001 # don't split if entropy reduction is less than this
    MAX_DEPTH = 10
    index = 0

    def __init__(self, data=None, config=None, depth=0):
        self.config = config
        self.depth = depth
        if config != None:
            self.N_THRESHOLD = config[0]
            self.H_THRESHOLD = config[1]
            self.H_REDUCTION_THRESHOLD = config[2]
            self.MAX_DEPTH = config[3]
        
        DTNode.index += 1
        self.index = DTNode.index # node has unique number
        self.data = data
        self.prob = None
        if data is not None:
            self.n = float(data.shape[0]) # number of examples
            self.indices = range(data.shape[1] - 1) # feature indices
            self.set_h()

        self.splits = {}

        self.feat_id = None # feature index
        self.thres = None # threshold
        self.lchild = None # left child
        self.rchild = None # right child
        self.parent = None

    # Create split on feature 'i' at value 'th'
    def split(self, i, th):
        self.feat_id = i
        self.thres = th
        self.lchild = DTNode(self.data[self.data[:, i] < th], self.config)
        self.rchild = DTNode(self.data[self.data[:, i] >= th], self.config)
        self.splits[i].remove(th)

    # Evaluate candidate split by weighted average entropy
    def split_eval(self, i, th):
        lc = DTNode(self.data[self.data[:, i] < th], self.config, self.depth + 1)
        rc = DTNode(self.data[self.data[:, i] >= th], self.config, self.depth + 1)
        pl = lc.n / self.n
        pr = 1.0 - pl
        avgH = pl * lc.H + pr * rc.H
        return avgH, lc, rc
    
    # Entropy of class labels in this node, assumes 1, -1
    def set_h(self):
        b = .001
        npos = np.sum(self.data[:,-1] == 1) # count labels 1
        p = (npos + b) / (self.n + b + b)
        self.prob = p
        self.H = -p * np.log(p) - (1-p) * np.log(1-p)

    def build_tree(self):
        if self.H < self.H_THRESHOLD or self.n <= self.N_THRESHOLD:
            return
        if self.depth >= self.MAX_DEPTH:
            return
        # Find the best split
        (i, th, (h, lc, rc)) = argmax([(i, th, self.split_eval(i, th)) \
                                            for i in self.indices \
                                            for th in self.get_splits(i)],
                                        lambda x : -x[2][0]) # x = (a, b, (h, c, d))
        
        if (self.H - h) < self.H_REDUCTION_THRESHOLD:
            return
        
        
        # Recurse
        self.feat_id = i
        self.thres = th
        self.lchild = lc
        self.rchild = rc
        self.lchild.parent = self
        self.rchild.parent = self
        self.lchild.build_tree()
        self.rchild.build_tree()
    
    # Sort examples and return middle points between every two consecutive samples
    def get_splits(self, i):
        if i not in self.splits:
            # d = np.sort(np.unique(self.data[:,i]), axis=None)
            # d1 = d[:-1]
            # d2 = d[1:]
            # self.splits[i] = (d1 + d2) / 2.0
            self.splits[i] = np.sort(np.unique(self.data[:,i]), axis=None)
        return self.splits[i]

    # Classify a data point
    def classify(self, x):
        if self.feat_id == None: # leaf node
            return self.prob
        elif x[self.feat_id] < self.thres:
            return self.lchild.classify(x) # go to left child
        else:
            return self.rchild.classify(x) # go to right child
        
    def display(self, depth=0, max_depth=3):
        if depth > max_depth:
            print(depth*'  ', 'Depth >', max_depth)
        if self.feat_id is None:
            print(depth*'  ', '=>', '%.2f'%self.prob, '[ n=', self.n, ']')
            return
        print(depth*'  ', 'Ft.', self.feat_id, '<', self.thres, '[ n=', self.n, ']')
        self.lchild.display(depth+1, max_depth)
        self.rchild.display(depth+1, max_depth)

def argmax(l, f):
    """
    Return the element in list l that gives highest value on f

    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

class DecisionTree:
    def fit(self, X, Y, config=None):
        D = np.hstack([X,Y])
        self.root = DTNode(D, config)
        self.root.build_tree()
    def predict(self, X):
        pred = np.array([np.apply_along_axis(self.root.classify, 1, X)]).T - 0.5
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred
    def display(self, depth=0, max_depth=3):
        self.root.display(depth, max_depth)

class RandomForest:
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