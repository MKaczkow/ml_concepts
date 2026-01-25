# GPT From Scratch

## LayerNorm vs BatchNorm
* `Batch Normalization` normalizes each feature across batch dimension, which means 'for each batch, we expect the features to have zero mean and unit variance'
* `Layer Normalization` normalizes each feature across the feature dimension, which means 'for each sample, we expect the features to have zero mean and unit variance'
* ... so layer norm helps with vanishing / exploding gradient

## Pre-LayerNorm vs Post-LayerNorm
* `Pre-LayerNorm` is applied before the attention and feedforward layers and dropout is applied after them
* `Post-LayerNorm` has been used in older architectures, such as the original Transformer model

## Chapter 7
* finetuning for instruction following
* instruction (instruction) -> desired response (output)
* mask placeholder tokens in the output, by adding like *-100* at the end of sequence (the exact value is not random - it's because cross-entropy loss ignores targets with value -100, with `ignore_index=-100` argument, which is default in PyTorch)
* custom `collate` functions pads each sequence in the batch to the same length (max length in the batch), *but* allows for different lengths across batches
* usage of `functools.partial` to create a new function with some arguments fixed