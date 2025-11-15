import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,precision_score, recall_score, f1_score
import time



#---------------Softmax Logistic Regression-----------------
class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None    
        self.b = None    

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def _cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = np.max(y) + 1 


        self.W = np.random.randn(num_features, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))


        Y_onehot = self._one_hot(y, num_classes)


        for epoch in range(self.epochs):

            logits = np.dot(X, self.W) + self.b
            probs = self._softmax(logits)

            loss = self._cross_entropy_loss(Y_onehot, probs)

            grad_logits = (1./ num_samples) * (Y_onehot - probs) 
            grad_W = -np.dot(X.T, grad_logits)
            grad_b = -np.sum(grad_logits, axis=0, keepdims=True)


            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b


    def predict_proba(self, X):
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    





#--------------Decision Tree Node------------------
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

#-----------------Approximate method for XGBoost-------------------
class XGBoostTree_approx:
    def __init__(self, max_depth=3, n_bins=10, lam=1):
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.lam = lam
        self.root = None

    def fit(self, X, g, h):
        self.root = self.build_tree(X, g, h, depth=0)

    def build_tree(self, X, g, h, depth):
        n_samples, n_features = X.shape

        # Leaf stopping condition
        if depth >= self.max_depth:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lam)
            return DecisionTreeNode(value=leaf_value)

        best_gain = -float("inf")
        best_feat, best_thresh = None, None

        # Loop over ALL features
        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]

            # Skip constant features
            if np.all(feature_values == feature_values[0]):
                continue

            # Create bins
            bins = np.linspace(np.min(feature_values), np.max(feature_values), self.n_bins + 1)
            bin_ids = np.digitize(feature_values, bins) - 1

            g_bin = np.zeros(self.n_bins)
            h_bin = np.zeros(self.n_bins)

            for b in range(self.n_bins):
                mask = bin_ids == b
                if np.any(mask):
                    g_bin[b] = np.sum(g[mask])
                    h_bin[b] = np.sum(h[mask])

            g_cumsum = np.cumsum(g_bin)
            h_cumsum = np.cumsum(h_bin)

            g_total = g_cumsum[-1]
            h_total = h_cumsum[-1]

            # Evaluate possible splits at each bin boundary
            for b in range(1, self.n_bins):
                GL, HL = g_cumsum[b - 1], h_cumsum[b - 1]
                GR, HR = g_total - GL, h_total - HL

                if HL <= 0 or HR <= 0:
                    continue

                gain = 0.5 * (
                    (GL ** 2) / (HL + self.lam) +
                    (GR ** 2) / (HR + self.lam) -
                    (g_total ** 2) / (h_total + self.lam)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = bins[b]

        # If no valid split → make leaf
        if best_feat is None:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lam)
            return DecisionTreeNode(value=leaf_value)

        # Partition data
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        left_subtree = self.build_tree(X[left_idx], g[left_idx], h[left_idx], depth + 1)
        right_subtree = self.build_tree(X[right_idx], g[right_idx], h[right_idx], depth + 1)

        return DecisionTreeNode(best_feat, best_thresh, left_subtree, right_subtree)

    def predict(self, X):
        return np.array([self._predict_row(x, self.root) for x in X])

    def _predict_row(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)




#--------------Multiclass XGBoost Classifier (Softmax)---------------------
class XGBoostClassifierMulticlass:
    def __init__(self, n_estimators=20, learning_rate=0.3,
                 max_depth=3, n_bins=10, lam=1):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.lam = lam
        self.trees = []  # trees[round][class]

    def softmax(self, scores):
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples = len(y)
        self.K = len(np.unique(y))

        # One-hot encoded targets
        y_onehot = np.eye(self.K)[y]

        # Initial scores
        y_pred = np.zeros((n_samples, self.K))

        for _ in range(self.n_estimators):

            # Probabilities from softmax
            P = self.softmax(y_pred)

            # First-order gradient
            g = P - y_onehot

            # Second-order (diagonal Hessian approx)
            h = P * (1 - P)

            trees_round = []

            # Train one tree per class
            for k in range(self.K):
                tree = XGBoostTree_approx(
                    max_depth=self.max_depth,
                    n_bins=self.n_bins,
                    lam=self.lam
                )

                tree.fit(X, g[:, k], h[:, k])
                trees_round.append(tree)

                # Update model output
                y_pred[:, k] += self.lr * tree.predict(X)

            self.trees.append(trees_round)

    def predict_proba(self, X):
        n = X.shape[0]
        scores = np.zeros((n, self.K))

        # Sum boosted predictions
        for trees_round in self.trees:
            for k in range(self.K):
                scores[:, k] += self.lr * trees_round[k].predict(X)

        return self.softmax(scores)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


#------------Histogram-based XGBoost Tree with Random Feature Selection-------------
class XGBoostTree_approx_randomforest:
    def __init__(self, max_depth=3, n_bins=10, lam=1, feature_subsample_size=None):
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.lam = lam
        self.feature_subsample_size = feature_subsample_size
        self.root = None

    def fit(self, X, g, h):
        self.root = self.build_tree(X, g, h, depth=0)

    def build_tree(self, X, g, h, depth):
        n_samples, n_features = X.shape

        # Leaf node condition
        if depth >= self.max_depth:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lam)
            return DecisionTreeNode(value=leaf_value)

        # Random feature subset
        if self.feature_subsample_size is None:
            m = int(np.sqrt(n_features))
        else:
            m = min(self.feature_subsample_size, n_features)

        feature_indices = np.random.choice(n_features, m, replace=False)

        best_gain = -float("inf")
        best_feat, best_thresh = None, None

        # Search best split among chosen features
        for feat_idx in feature_indices:
            feature_values = X[:, feat_idx]

            # Skip if constant feature
            if np.all(feature_values == feature_values[0]):
                continue

            # Binning
            bins = np.linspace(np.min(feature_values), np.max(feature_values), self.n_bins + 1)
            bin_ids = np.digitize(feature_values, bins) - 1

            g_bin = np.zeros(self.n_bins)
            h_bin = np.zeros(self.n_bins)

            for b in range(self.n_bins):
                mask = bin_ids == b
                if np.any(mask):
                    g_bin[b] = np.sum(g[mask])
                    h_bin[b] = np.sum(h[mask])

            g_cumsum = np.cumsum(g_bin)
            h_cumsum = np.cumsum(h_bin)

            g_total = g_cumsum[-1]
            h_total = h_cumsum[-1]

            # Evaluate split gain for each bin
            for b in range(1, self.n_bins):
                GL, HL = g_cumsum[b - 1], h_cumsum[b - 1]
                GR, HR = g_total - GL, h_total - HL

                if HL <= 0 or HR <= 0:
                    continue

                gain = 0.5 * (
                    (GL ** 2) / (HL + self.lam) +
                    (GR ** 2) / (HR + self.lam) -
                    (g_total ** 2) / (h_total + self.lam)
                )

                if np.isnan(gain) or np.isinf(gain):
                    continue

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = bins[b]

        # If no split found → leaf
        if best_feat is None:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lam)
            return DecisionTreeNode(value=leaf_value)

        # Split data
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        left_subtree = self.build_tree(X[left_idx], g[left_idx], h[left_idx], depth + 1)
        right_subtree = self.build_tree(X[right_idx], g[right_idx], h[right_idx], depth + 1)

        return DecisionTreeNode(best_feat, best_thresh, left_subtree, right_subtree)

    def predict(self, X):
        return np.array([self._predict_row(x, self.root) for x in X])

    def _predict_row(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)



#--------------------Multiclass XGBoost Classifier with RandomForest-style trees----------------------

class XGBoostClassifierMulticlass_RandomForest:
    def __init__(self, n_estimators=20, learning_rate=0.3,
                 max_depth=3, n_bins=10, lam=1,
                 feature_subsample_size=None):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.lam = lam
        self.feature_subsample_size = feature_subsample_size
        self.trees = []        # list of lists: trees[round][class]

    # Softmax
    def softmax(self, scores):
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples = len(y)
        self.K = len(np.unique(y))

        # One-hot encode y
        y_onehot = np.eye(self.K)[y]

        # Initial prediction scores = zeros
        y_pred = np.zeros((n_samples, self.K))

        for _ in range(self.n_estimators):

            # Compute softmax probabilities
            P = self.softmax(y_pred)

            # Multiclass gradient and diagonal Hessian
            g = (P - y_onehot)
            h = P * (1 - P)

            trees_round = []

            # Train K trees for K classes
            for k in range(self.K):
                tree = XGBoostTree_approx_randomforest(
                    max_depth=self.max_depth,
                    n_bins=self.n_bins,
                    lam=self.lam,
                    feature_subsample_size=self.feature_subsample_size
                )

                tree.fit(X, g[:, k], h[:, k])
                trees_round.append(tree)

                # Update scores
                y_pred[:, k] += self.lr * tree.predict(X)

            self.trees.append(trees_round)

    # Predict class probabilities
    def predict_proba(self, X):
        n = X.shape[0]
        scores = np.zeros((n, self.K))

        for trees_round in self.trees:
            for k in range(self.K):
                scores[:, k] += self.lr * trees_round[k].predict(X)

        return self.softmax(scores)

    # Class labels
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


#------------Multiclass XGBoost Classifier with RandomForest and applying One vs Rest---------------------
#--------------------Histogram Tree with Random Feature Subsampling------------------------
class XGBoostTree_approx_randomforest:
    def __init__(self, max_depth=3, n_bins=10, lam=1, feature_subsample_size=None):
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.lam = lam
        self.feature_subsample_size = feature_subsample_size
        self.root = None

    def fit(self, X, g, h):
        self.root = self.build_tree(X, g, h, depth=0)

    def build_tree(self, X, g, h, depth):
        n_samples, n_features = X.shape

        if depth >= self.max_depth:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lam)
            return DecisionTreeNode(value=leaf_value)

        # Random subset of features (RF-style)
        if self.feature_subsample_size is None:
            m = int(np.sqrt(n_features))
        else:
            m = min(self.feature_subsample_size, n_features)

        feature_indices = np.random.choice(n_features, m, replace=False)

        best_gain = -float("inf")
        best_feat, best_thresh = None, None

        for feat_idx in feature_indices:
            feature_values = X[:, feat_idx]

            if np.all(feature_values == feature_values[0]):
                continue

            bins = np.linspace(np.min(feature_values), np.max(feature_values), self.n_bins + 1)
            bin_ids = np.digitize(feature_values, bins) - 1

            g_bin = np.zeros(self.n_bins)
            h_bin = np.zeros(self.n_bins)

            for b in range(self.n_bins):
                mask = bin_ids == b
                if np.any(mask):
                    g_bin[b] = np.sum(g[mask])
                    h_bin[b] = np.sum(h[mask])

            g_cumsum = np.cumsum(g_bin)
            h_cumsum = np.cumsum(h_bin)

            g_total = g_cumsum[-1]
            h_total = h_cumsum[-1]

            for b in range(1, self.n_bins):
                GL, HL = g_cumsum[b - 1], h_cumsum[b - 1]
                GR, HR = g_total - GL, h_total - HL

                if HL <= 0 or HR <= 0:
                    continue

                gain = 0.5 * (
                    (GL ** 2) / (HL + self.lam) +
                    (GR ** 2) / (HR + self.lam) -
                    (g_total ** 2) / (h_total + self.lam)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = bins[b]

        if best_feat is None:
            leaf_value = -np.sum(g) / (np.sum(h) + self.lam)
            return DecisionTreeNode(value=leaf_value)

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        left_subtree = self.build_tree(X[left_idx], g[left_idx], h[left_idx], depth + 1)
        right_subtree = self.build_tree(X[right_idx], g[right_idx], h[right_idx], depth + 1)

        return DecisionTreeNode(best_feat, best_thresh, left_subtree, right_subtree)

    def predict(self, X):
        return np.array([self._predict_row(x, self.root) for x in X])

    def _predict_row(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)



#--------------------OvR Multiclass XGBoost (using sigmoid)------------------------
class XGBoostClassifierMulticlass_OvR_RF:
    def __init__(self, n_estimators=20, learning_rate=0.3,
                 max_depth=3, n_bins=10, lam=1, feature_subsample_size=None):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.lam = lam
        self.feature_subsample_size = feature_subsample_size
        self.trees = []   # trees[class][round]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples = len(y)
        self.K = len(np.unique(y))

        # Store K sets of boosted trees
        self.trees = [[] for _ in range(self.K)]

        # Train K binary classifiers (one-vs-rest)
        for k in range(self.K):

            # Create binary labels for class k
            y_binary = (y == k).astype(float)

            # Initial prediction scores
            y_pred = np.zeros(n_samples)

            for _ in range(self.n_estimators):

                p = self.sigmoid(y_pred)
                g = p - y_binary          # gradient
                h = p * (1 - p)           # Hessian

                tree = XGBoostTree_approx_randomforest(
                    max_depth=self.max_depth,
                    n_bins=self.n_bins,
                    lam=self.lam,
                    feature_subsample_size=self.feature_subsample_size
                )

                tree.fit(X, g, h)
                self.trees[k].append(tree)

                # Update predictions
                y_pred += self.lr * tree.predict(X)

    def predict_proba(self, X):
        n = X.shape[0]
        scores = np.zeros((n, self.K))

        # Compute probability from each OvR classifier
        for k in range(self.K):
            raw = np.zeros(n)
            for tree in self.trees[k]:
                raw += self.lr * tree.predict(X)
            scores[:, k] = self.sigmoid(raw)

        return scores

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)