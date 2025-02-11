# Style Transfer
... for images.

## Idea
* components
  * `original image`
  * `style image`
  * `generated image` (initialized as noise)
* ... so we change input instead of weights (they are frozen)

## Loss Function
$\mathcal{L}_{total}(G) = \alpha \mathcal{L}_{content}(C, G) + \beta \mathcal{L}_{style}(S, G)$  
* combines `content loss` and `style loss`

### Content Loss
$\mathcal{L}_{content}(C, G) = \frac{1}{2} \sum_{i, j} (F_{ij}^C - F_{ij}^G)^2$
* basically, takes norm for every selected layers' outputs for `content` and `generated` images
* `Gram matrix`, which is a matrix multiplication of the feature map with its transpose  

## References
* [yt tutorial](https://www.youtube.com/watch?v=imX4kSKDY7s&ab_channel=AladdinPersson)
