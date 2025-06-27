# Kernels

# TODO
- [x] tf is kernel perceptron?

## General
* you can just add a new dimension if data is not linearly separable in current dimensions
* you can pretend to have more dimensions than you actually have, using `kernels`
* `kernel` is similarity function that measures how similar two data points are
* they are done using `inner product` (*how much overlap do two vectors have?*)
* `kernel trick` - a method used in machine learning algorithms to operate in a high-dimensional space without explicitly mapping data points to that space, which is computationally expensive
* `RBF (Radial Basis Function)` is default kernel, most of the time will work
* ... and also works in infinite dimensions
```math
K(x, y) = e^{-\gamma ||x - y||^2}
```
* `polynomial kernel` - kernel which have `d` parameter (deegree of polynomial), which is used to control the complexity of the decision boundary
```math
K(x, y) = (x \cdot y + c)^d
```

## References
* [kernel overview yt sentdex](https://www.youtube.com/watch?v=9IfT8KXX_9c&ab_channel=sentdex)
* [kernel motivation yt sentdex](https://www.youtube.com/watch?v=xqg5S-GrrDQ&ab_channel=sentdex)
* SVM + kernels statquest [part 1](https://www.youtube.com/watch?v=efR1C6CvhmE&ab_channel=StatQuestwithJoshStarmer) [part 2](https://www.youtube.com/watch?v=Toet3EiSFcM&ab_channel=StatQuestwithJoshStarmer) [part 3](https://www.youtube.com/watch?v=Qc5IyLW_hns&ab_channel=StatQuestwithJoshStarmer)
