Run the following commands to run using  `mps` framework:

``` 
conda create -n linter python==3.9
pip install pre-commit csbdeep tifffile einops pyyaml pandas matplotlib imagecodecs scikit-image
pip install torch torchvision
```

TODO: Configs to figure out about original experiments in [Latent Diffusion](https://github.com/CompVis/latent-diffusion/tree/main):
- [ ] Batch size 
- [ ] Learning Rate
- [ ] Optimizer
- [ ] Number of Iterations
