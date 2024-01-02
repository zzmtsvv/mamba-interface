## Mamba clean code in jax and PyTorch

Actually, this is my one-evening attempt to get more handy with `jax` and `flax` on the basis of `torch`
implementation on the example of Mamba[1]. It looks more like a somewhat detailed interface of this model that also requires training and inference code. I hope this code will help you become more confident with 
jax, flax or state-space models[2].

Feel free to contact me on any mistakes you find :)
Unfortunately, my code lacks [associative scan](https://arxiv.org/abs/1709.04057) (yet)

This repo is based on the following ones: [annotated-mamba](https://github.com/srush/annotated-mamba), [mamba-minimal in torch](https://github.com/johnma2006/mamba-minimal), [the official implementation](https://github.com/state-spaces/mamba)

## References
[1] - Gu, Dao et al. (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces.](https://arxiv.org/abs/2312.00752) <br/>
[2] Gu et al. (2022). [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
