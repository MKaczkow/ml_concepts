# GPT From Scratch

## LayerNorm vs BatchNorm
* `Batch Normalization` normalizes each feature across batch dimension, which means 'for each batch, we expect the features to have zero mean and unit variance'
* `Layer Normalization` normalizes each feature across the feature dimension, which means 'for each sample, we expect the features to have zero mean and unit variance'
* ... so layer norm helps with vanishing / exploding gradient

## Pre-LayerNorm vs Post-LayerNorm
* `Pre-LayerNorm` is applied before the attention and feedforward layers and dropout is applied after them
* `Post-LayerNorm` has been used in older architectures, such as the original Transformer model
