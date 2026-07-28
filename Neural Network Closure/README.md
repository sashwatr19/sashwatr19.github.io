# Neural Network Closure Model for Burgers' Equation

A neural closure model that corrects a coarse grid solver of the Burgers equation so that it tracks a high-resolution reference. Follows the setup of Melchers et al. (2023).

Solving the equation on a fine grid is accurate but expensive while solving it on a coarse grid is cheap but loses physical information. This project trains a convolutional neural network to learn the closure term, the lost physical information, so that a coarse solver plus the network output reproduces the downsampled fine-grid solution at a fraction of the cost.

## Approach

- Fine-grid (4096-point) Burgers trajectories are generated from random initial conditions and averaged down to a 64-point coarse grid to produce the reference solution `u_ref` and its time derivative `dudt_ref`. A plain coarse solve `u_coarse` is also generated for comparison. The PyTorch version (`data_gen_torch.py`) solves all trajectories simultaneously as one batched ODE for speed.
- A 1D convolutional neural network is designed with the physics built in, it wraps around at the boundaries to match the periodic domain, takes both `u` and `u²` as input, and outputs its correction as a difference between neighbouring points to satisfy mass conservation. This network is added to the coarse solver's update rule and trained to match the accurate rate of change measured from the fine-grid data.
- The trained closure model is placed inside an ODE solver and rolled out from held-out test initial conditions, comparing the corrected coarse trajectory against the reference.

## Repository contents

- `burgers.ipynb` -  model definition, training, and rollout/evaluation.
- `data_gen.py` - dataset generation (SciPy, sequential).
- `data_gen_torch.py` - dataset generation (PyTorch + torchdiffeq, batched).
- `burgers.npz` - generated dataset (`u_ref`, `dudt_ref`, `u_coarse`).
- `closure_cnn_derivative*.pt` - trained model weights.
- `losses.npy` - training loss history.

## Getting started

Generate the dataset (or use the included `burgers.npz`):
```
python data_gen_torch.py
```

## Usage

Use the notebook `burgers.ipynb` to train and evaluate:
```
jupyter notebook burgers.ipynb
```

## Dependencies
- NumPy
- SciPy
- Matplotlib
- PyTorch
- torchdiffeq (for the batched data generator)
- tqdm

## Reference
Melchers et al. (2023), on neural closure models for coarse-grained PDE solvers.
