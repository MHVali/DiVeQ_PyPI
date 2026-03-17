# DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick

This is the code implementation for the paper [*"DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick"*](https://arxiv.org/abs/2509.26469) accepted at ICLR 2026.

**Abstract:**
Vector quantization is common in deep models, yet its hard assignments block gradients and hinder end-to-end training. We propose DiVeQ, which treats quantization as adding an error vector that mimics the quantization distortion, keeping the forward pass hard while letting gradients flow. We also present a space-filling variant (SF-DiVeQ) that assigns to a curve constructed by the lines connecting codewords, resulting in less quantization error and full codebook usage. Both methods train end-to-end without requiring auxiliary losses or temperature schedules. In VQ-VAE image compression, VQGAN image generation, and DAC speech coding tasks across various data sets, our proposed methods improve reconstruction and sample quality over alternative quantization approaches.

![alt text](https://raw.githubusercontent.com/AaltoML/DiVeQ/main/diveq_teaser.png)

# Quickstart
Check the quickstart guide [here](https://MHVali.github.io/diveq/).

# Documentation
The `diveq` documentation is available online [here](https://MHVali.github.io/diveq/). You can also view it locally by running:
```bash
mkdocs serve

## Citation

```
@InProceedings{vali2026diveq,
    title={{DiVeQ}: {D}ifferentiable {V}ector {Q}uantization {U}sing the {R}eparameterization {T}rick},
    author={Vali, Mohammad Hassan and Bäckström, Tom and Solin, Arno},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026}
}
```

## License
This software is provided under the MIT License. See the accompanying [LICENSE](LICENSE.txt) file for details.