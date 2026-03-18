# DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick

This is the code implementation for the paper [*"DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick"*](https://arxiv.org/abs/2509.26469) accepted at ICLR 2026.

**Abstract:**
Vector quantization is common in deep models, yet its hard assignments block gradients and hinder end-to-end training. We propose DiVeQ, which treats quantization as adding an error vector that mimics the quantization distortion, keeping the forward pass hard while letting gradients flow. We also present a space-filling variant (SF-DiVeQ) that assigns to a curve constructed by the lines connecting codewords, resulting in less quantization error and full codebook usage. Both methods train end-to-end without requiring auxiliary losses or temperature schedules. In VQ-VAE image compression, VQGAN image generation, and DAC speech coding tasks across various data sets, our proposed methods improve reconstruction and sample quality over alternative quantization approaches.

![alt text](https://raw.githubusercontent.com/AaltoML/DiVeQ/main/diveq_teaser.png)

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

# Installation

## Install using pip
The easiest way to install `diveq` is through `pip` by running:

```bash
pip install diveq
```

After installing `diveq` you can verify the installation by running:

```bash
diveq --version
```

This should output:

```bash
diveq version x.y.z yyyy-zzzz developed by Mohammad Vali (AaltoML Research Group, Aalto University)
```

Where:

- `x.y.z` represents the major, minor, and patch version.
- `yyyy-zzzz` indicates the development start year and the current year.

## Install using uv 

`uv` is a modern python package manager. You can see more details about `uv` in [the official documentation](https://docs.astral.sh/uv/).

First, you need to install `uv` and `uvx` following the instructions for your operating system in <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank">`uv` website</a>.

Then run:

```bash
uv tool install diveq
```

You can verify the installation running:

```bash
uv tool run diveq --version
```

or you can use the shortcut version `uvx`:

```bash
uvx diveq --version
```

This should output:

```bash
diveq version x.y.z yyyy-zzzz developed by Mohammad Vali (AaltoML Research Group, Aalto University)
```

Where:

- `x.y.z` represents the major, minor, and patch version.
- `yyyy-zzzz` indicates the development start year and the current year.

# Usage Example

Before using `diveq`, you have to install it by one of the methods mentioned earlier.

## Use DiVeQ for vector quantization in a VQ-VAE
In **example** directory, we provide a code example of how `diveq` is used in a vector quantized variational autoencoder (VQ-VAE). A minimal example is shown below:

```
from diveq import DIVEQ
.
.
.
self.vq = DIVEQ(num_embeddings, embedding_dim)
```

In this example:

- `self.vq` is the vector quantization module that will be used for buidling the model.
- `num_embeddings` and `embedding_dim` are the codebook size and dimension of each codebook entry, respectively. In the following, you can find the list of all parameters used in different quantization methods incorporated in `diveq` package.

## List of parameters
Here, we provide the list of parameters that are used as inputs to eight different vector quantization methods included in `diveq` package.

- `num_embeddings` (integer): Codebook size or the number of codewords in the codeook.
- `embedding_dim` (integer): Dimensionality of each codebook entry or codeword.
- `noise_var` (float): Variance of the directional noise for *DiVeQ*-based methods.
- `replacement_iters` (integer): Number of training iterations to apply codebook replacement.
- `discard_threshold` (float): Threshold to discard the codebook entries that are used less than this threshold after *replacement_iters* iterations.
- `perturb_eps` (float): Adjusts perturbation/shift magnitude from used codewords for codebook replacement.
- `uniform_init` (bool): Whether to use uniform initialization. If False, codebook is initialized from a normal distribution.
- `verbose` (bool): Whether to print codebook replacement status, i.e., to print how many unused codewords are replaced.
- `skip_iters` (integer): Number of training iterations to skip quantization (for *SF-DiVeQ* and *SF-DiVeQ_Detach*) or to use *DiVeQ* quantization (for *Residual_SF-DiVeQ* and *Product_SF-DiVeQ*) to have a custom initialization.
- `avg_iters` (integer): Number of recent training iterations to extract latents for custom codebook initialization in Space-Filling Versions.
- `latents_on_cpu` (bool): Whether to collect latents for custom initialization on cpu. If running out of CUDA memory, set it to True.
- `allow_warning` (bool): Whether to print the warnings. The warnings will warn if the user inserts unusual values for the parameters.
- `num_codebooks` (integer): Number of codebooks to be used for quantization in VQ variants of Residual VQ and Product VQ. All the codebooks will have the same size and dimensionality.

# Important Notes

1. **Codebook Replacement:** Note that to prevent *codebook collapse*, we include a codebook replacement function (in cases where it is required) inside different quantization modules. Codebook replacement function is called after each `replacement_iters` training iterations, and it replaces the codewords which are used less than `discard_threshold` with a pertubation of an actively used codeword which is shifted by `perturb_eps` magnitude. If `verbose=True`, the status of how many unsued codewords is replaced will be printed by the module. Note that the number of unused codewords should be decreased over training and it might take a while.

2. **Variants of Vector Quantization:** Residual VQ and Product VQ are two variants of vector quantization which are included in the `diveq` package. These variants utilize multiple codebooks to perform the quantization, where `num_codebooks` determinses the number of codebooks used in these VQ variants.

3. **Space-Filling Methods:** Quantization methods based on Space-Filling (i.e., *SF-DiVeQ*, *SF-DiVeQ_Detach*, *Residual_SF-DiVeQ*, *Product_SF-DiVeQ*) use a custom initilization. *SF-DiVeQ* and *SF-DiVeQ_Detach* skip quantizing the latents for `skip_iters` training iterations, and initialize the codebook with an average of latents captured from `avg_iters` recent training iterations. After this custom initialization, they start to quantize the latents. *Residual_SF-DiVeQ* and *Product_SF-DiVeQ* work in the same way, but they apply *DiVeQ* for the first `skip_iters` training iterations. Note that if `avg_iters` value is set to a large value, CUDA might run out of memory, as there should be a large pull of latents to be stored for custom initialization. Therefore, user can set `latents_on_cpu=True` to store the latents on CPU, or set a smaller value for `avg_iters`.

4. **Detach Methods:** *DiVeQ_Detach* and *SF-DiVeQ_Detach* methods do not use directional noise. Therefore, they do not need to set the `noise_var` parameter.

For further details about different vector quantization methods in the `diveq` package and their corresponding parameters, please see the details provided in the main python codes.

# Citation

```
@InProceedings{vali2026diveq,
    title={{DiVeQ}: {D}ifferentiable {V}ector {Q}uantization {U}sing the {R}eparameterization {T}rick},
    author={Vali, Mohammad Hassan and Bäckström, Tom and Solin, Arno},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026}
}
```

# License
This software is provided under the MIT License. See the accompanying [LICENSE](LICENSE.txt) file for details.