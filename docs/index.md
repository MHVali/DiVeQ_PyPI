# Welcome to diveq
`diveq` (short for differentiable vector quantization) is a tool designed to implement and train vector quantization (VQ) in deep neural networks (DNNs), like a VQ-VAE. It allows end-to-end training of DNNs which contain the non-differentiable VQ module, without any auxiliary loss and hyperparameter tunings. `diveq` is implemented via PyTorch and it requires `python >= 3.11` and `torch >= 1.13`.

`diveq` method is published as a research paper entitled [*"DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick"*](https://arxiv.org/abs/2509.26469) at International Conference on Learning Representations (ICLR) 2026.

`diveq` package includes eight different quantization methods:
1. `from diveq import DIVEQ` optimizes the VQ codebook via DiVeQ technique
2. `from diveq import SFDIVEQ` optimizes the VQ codebook via SF-DiVeQ technique
3. `from diveq import DIVEQDetach` optimizes the VQ codebook via DiVeQ_Detach technique
4. `from diveq import SFDIVEQDetach` optimizes the VQ codebook via SF-DiVeQ_Detach technique

Other VQ variants that uses multiple codebooks for quantization
5. `from diveq import ResidualDIVEQ` optimizes the VQ codebook via Residual_DiVeQ technique
6. `from diveq import ResidualSFDIVEQ` optimizes the VQ codebook via Residual_SF-DiVeQ technique
7. `from diveq import ProductDIVEQ` optimizes the VQ codebook via Product_DiVeQ technique
8. `from diveq import ProductSFDIVEQ` optimizes the VQ codebook via Product_SF-DiVeQ technique

## Guides
To begin using `diveq`, refer to the guides below:

- [Installation](install.md)
- [Quickstart](quickstart.md)

For a more detailed description of individual features, see the [Reference](reference.md) section.
