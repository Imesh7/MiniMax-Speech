# MiniMax-Speech

A TTS model support voice clonning & emotions

paper - https://arxiv.org/abs/2505.07916


# Neural codec or Audio tokenizer

Audio tokenizer is Encoder-VQ-Decoder based

# Auto-Regressive transformer

This is called as Speech Learnable encoder in paper


# Flow matching

have 2 components

## Flow-VAE

Flow based Autoencoder(Encoder & decoder) architecture used to implemented it.
`AudioFlow` class, firstly used define `prior`; prior is a normal distribution (torch.distributions.normal.Normal). 
It has a coupling layers (few ouplig layers) it will iterate through those layers.

While sampling those couplig layers will reverse as well.

`CoupligLayerFlow` class is used to implement normalizing flows.
It is used scalling & adaptive translator layers

## Flow matching model

This is used to estimate the velocity for the flows.


### Specially you should know

The difference between 'Autoregressive transformer' vs 'Non-autoregressive transformer'

Autoregressive transformer's output will depends only on previous tokens.