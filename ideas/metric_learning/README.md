# Metric learning

### General
* define a metric that reflects the similarity between data
* need for good representation of data in high-dimensional space
* similar data == close in metric space
* `Mahanalobis distance` - euclidean distance in transformed space, defined by matrix M
`D(x, x') = sqrt((Lx - Lx')^T * (Lx - Lx'))`

### Metric space
* `metric space` must satisfy 4 properties:
    - `distance from point to itself is 0` - `d(x, x) = 0`
    - `symetry` - `d(x, y) = d(y, x)`
    - `triangle inequality` - `d(x, y) + d(y, z) >= d(x, z)`
    - `positivity` - `d(x, y) >= 0`

### Some examples
* `LMNN: Large Margin Nearest Neighbor` - *knn-like* algorithm which learns pseudo-metric similar to the one used in kNN
* `ITML: Information Theoretic Metric Learning` - minimizes relative entropy (Kullback-Leibler divergence) between two distributions, which are assumed to be Gaussian and multivariate
* `LSML: Least Squares Metric Learning` - minimizes a convex objective function corresponding to the sum of squared distances between pairs of points
* `SDML: Sparse Determinant Metric Learning` - sparse metric learning, applying 2 types of regularization in high-dimensional space: L1 for off-diagonal elements of M matrix and log-det divergence 
