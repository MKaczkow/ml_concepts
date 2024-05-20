# Lipschitz Networks
* `Lipschitz network`
* In [this lecture](http://zbum.ia.pw.edu.pl/PL/dydaktyka/SNR/PUBLIC/SNR_2021Z_wyklad.pdf), something is mentioned in the context of estimating approximation capabilities.
* *Networks representing continuous-time dynamic systems* (Lecture 1, Slide 12)

## General
* Addressing the lack of `robustness` in neural networks, meaning neural networks are not like Java, as they are not robust.
* `Lipschitz continuity` is a property of a function to not change too rapidly
```plaintext
A function f : Rᴹ → Rᴺ is Lipschitz continuous if there is a constant L such that
∥f(x) - f(y)∥ ≦ L ∥x - y∥ for every x, y.
```
* `Spectral normalization` is a method that allows achieving `Lipschitz continuity` in neural networks.
* Partial solution to the problem of `adversarial examples`.
* activation functions
    - ReLU, Leaky ReLU, Softplus, Sigmoid, Tanh, ArcTan - `Lipschitz constant = 1`, meaning that derivatives are bounded by 1
    - `e^x` does not satisfy the Lipschitz condition
* Essentially, limiting the growth rate is limiting the derivative of the function (`gradient`)

## Resources
* [Towards Data Science article](https://towardsdatascience.com/lipschitz-continuity-and-spectral-normalization-b03b36066b0d)
* [OpenReview article](https://openreview.net/forum?id=BRIL0EFvTgc)
* [Repository](https://github.com/henrygouk/lipschitz-neural-networks)
* [arXiv: Unified Algebraic Perspective](https://arxiv.org/pdf/2303.03169)
* [NeurIPS: Regularity of Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2018/file/d54e99a6c03704e95e6965532dec148b-Paper.pdf)
* [Another paper](https://www.jmlr.org/papers/v25/22-1347.html)
* [NVIDIA: Learning Smooth Neural Functions via Lipschitz Regularization](https://nv-tlabs.github.io/lip-mlp/lipmlp_final.pdf)
* Posts on StackExchange Artificial Intelligence [1](https://ai.stackexchange.com/questions/30304/what-does-it-mean-having-lipschitz-continuous-derivatives) [2](https://ai.stackexchange.com/questions/1925/are-ffnn-mlp-lipschitz-functions) [3](https://ai.stackexchange.com/questions/29904/what-is-lipschitz-constraint-and-why-it-is-enforced-on-discriminator)
* [Wikipedia Lipschitz continuity](https://en.wikipedia.org/wiki/Lipschitz_continuity)
