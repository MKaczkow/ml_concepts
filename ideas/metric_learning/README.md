# Metric learning

* `LMNN: Large Margin Nearest Neighbor` - *knn-like* algorithm which learns pseudo-metric similar to the one used in kNN
* `ITML: Information Theoretic Metric Learning` - minimizes relative entropy (Kullback-Leibler divergence) between two distributions, which are assumed to be Gaussian and multivariate
* `LSML: Least Squares Metric Learning` - minimizes a convex objective function corresponding to the sum of squared distances between pairs of points
* `SDML: Sparse Determinant Metric Learning` - sparse metric learning, applying 2 types of regularization in high-dimensional space: L1 for off-diagonal elements of M matrix and log-det divergence 
