# MiniMax-Speech

A TTS model support voice clonning & emotions

paper - https://arxiv.org/abs/2505.07916


# Neural codec or Audio tokenizer

Audio tokenizer is Encoder-VQ-Decoder based

# Auto-Regressive transformer

This is called as Speech Learnable encoder in paper


# Flow matching

This will provide the voice cloning & emotion features to the model.
Have 2 components.

## Flow-VAE

Flow based Autoencoder(Encoder & decoder) architecture used to implemented it.
`AudioFlow` class, firstly used define `prior`; prior is a normal distribution (torch.distributions.normal.Normal). 

In training
It has a coupling layers (few couplig layers) it will iterate through those layers.


While sampling those couplig layers will reverse as well.

`CoupligLayerFlow` class (Affine Coupling Layers) 
is used to implement normalizing flows.
It is used scalling & adaptive translator layers 

## Flow matching model

This is used to estimate the velocity for the flows.


### Specially you should know

The difference between 'Autoregressive transformer' vs 'Non-autoregressive transformer'

Autoregressive transformer's output will depends only on previous tokens.


## Citation
Original Minimax-Speech paper:

```
@article{zhang2025minimaxspeech,
      title={MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder}, 
      author={Bowen Zhang and Congchao Guo and Geng Yang and Hang Yu and Haozhe Zhang and Heidi Lei and Jialong Mai and Junjie Yan and Kaiyue Yang and Mingqi Yang and Peikai Huang and Ruiyang Jin and Sitan Jiang and Weihua Cheng and Yawei Li and Yichen Xiao and Yiying Zhou and Yongmao Zhang and Yuan Lu and Yucen He},
      year={2025},
      eprint={2505.07916},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.07916}, 
}
```
