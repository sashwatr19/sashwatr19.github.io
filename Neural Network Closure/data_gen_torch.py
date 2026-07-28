import torch
import numpy as np
from torchdiffeq import odeint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

nu = 0.0005 # viscosity
x_start = 0.0 # domain start
x_final = 1.0 # domain end
x_range = x_final - x_start # domain range
t_final = 0.5 # simulation end time
num_fine_grid = 4096 # number of fine grid points
num_coarse_grid = 64 # number of coarse grid points
snap_timestep = 2**(-7) # snapshot interval
num_snapshots = 64 # number of additional snapshots
num_trajectories = 128 # total trajectories generated

dx_fine   = x_range / num_fine_grid
dx_coarse = x_range / num_coarse_grid


def burgers_rhs_torch(dx):
    """
    Returns the Burgers RHS function for a given grid spacing.
    Operates on batched input of shape (batch, nx).
    """
    def rhs(t, u):
        u_minus = torch.roll(u,  1, dims=1)
        u_plus  = torch.roll(u, -1, dims=1)
        term1 = (nu / dx**2) * (u_minus - 2*u + u_plus)
        alpha_plus  = 0.25 * torch.abs(u + u_plus)  - (1/12) * (u_plus  - u)
        alpha_minus = 0.25 * torch.abs(u_minus + u) - (1/12) * (u - u_minus)
        f_plus  = (1/6)*(u**2 + u*u_plus  + u_plus**2)  - alpha_plus  * (u_plus  - u)
        f_minus = (1/6)*(u_minus**2 + u_minus*u + u**2) - alpha_minus * (u - u_minus)
        return term1 - (f_plus - f_minus) / dx
    return rhs


def downsample_torch(u, factor=(num_fine_grid // num_coarse_grid)):
    """
    Downsample by block averaging.
    Input shape:  (batch, nx_fine)
    Output shape: (batch, nx_coarse)
    """
    batch, nx_fine = u.shape
    nx_coarse = nx_fine // factor
    return u.reshape(batch, nx_coarse, factor).mean(dim=2)

def make_initial_condition(num_grid_points, rng):

    """
    Creates initial conditions.
    """

    j = np.arange(num_grid_points) 
    u0 = np.zeros(num_grid_points, dtype=complex) # Initialize u0

    for k in range(1, 11):
        u_hat_pos = rng.standard_normal() + 1j * rng.standard_normal() # Independent gaussian distributed
        u_hat_neg = rng.standard_normal() + 1j * rng.standard_normal()  
        u0 += u_hat_pos * np.exp(2j * np.pi * k * j / num_grid_points) + u_hat_neg * np.exp(-2j * np.pi * k * j / num_grid_points) # Random addition of sine and cosine waves

    u0 = np.real(u0) # Restrict to real part
    u0 *= 2.0 / np.max(np.abs(u0)) # Rescale
    return u0


def generate_dataset_torch(seed=42):
    """
    Generate the full dataset of trajectories using torchdiffeq.
    All trajectories solved simultaneously in one batched ODE solve.

    Returns dict with:
      u_ref    : (num_trajectories, num_snapshots+1, num_coarse_grid)
      dudt_ref : (num_trajectories, num_snapshots+1, num_coarse_grid)
      u_coarse : (num_trajectories, num_snapshots+1, num_coarse_grid)
    """
    # --- Generate all initial conditions ---
    rng = np.random.default_rng(seed)
    u0_batch = np.stack([
        make_initial_condition(num_fine_grid, rng)
        for _ in range(num_trajectories)
    ])  # (128, 4096)

    u0_fine = torch.tensor(u0_batch, dtype=torch.float32).to(device)

    t_eval = torch.linspace(0, t_final, num_snapshots + 1).to(device)

    # --- Solve fine grid (all trajectories at once) ---
    print("Solving fine grid...")
    rhs_fine = burgers_rhs_torch(dx_fine)
    u_fine_traj = odeint(
        rhs_fine,
        u0_fine,
        t_eval,
        method='dopri5',       # adaptive RK45 equivalent
        rtol=1e-6,
        atol=1e-6,
    )  # shape: (num_snapshots+1, num_trajectories, num_fine_grid)

    # Permute to (num_trajectories, num_snapshots+1, num_fine_grid)
    u_fine_traj = u_fine_traj.permute(1, 0, 2)

    # --- Downsample to coarse grid ---
    print("Downsampling...")
    nt = u_fine_traj.shape[1]
    # Reshape to (num_trajectories * nt, num_fine_grid) for batched downsample
    u_fine_flat = u_fine_traj.reshape(-1, num_fine_grid)
    u_ref_flat  = downsample_torch(u_fine_flat)
    u_ref = u_ref_flat.reshape(num_trajectories, nt, num_coarse_grid)

    # --- Compute dudt_ref ---
    print("Computing derivatives...")
    rhs_fine_fn = burgers_rhs_torch(dx_fine)
    dudt_fine_flat = rhs_fine_fn(None, u_fine_flat)   # (num_trajectories*nt, num_fine_grid)
    dudt_ref_flat  = downsample_torch(dudt_fine_flat)  # (num_trajectories*nt, num_coarse_grid)
    dudt_ref = dudt_ref_flat.reshape(num_trajectories, nt, num_coarse_grid)

    # --- Solve coarse grid from downsampled ICs ---
    print("Solving coarse grid...")
    u0_coarse = u_ref[:, 0, :]   # (num_trajectories, num_coarse_grid)
    rhs_coarse = burgers_rhs_torch(dx_coarse)
    u_coarse_traj = odeint(
        rhs_coarse,
        u0_coarse,
        t_eval,
        method='dopri5',
        rtol=1e-6,
        atol=1e-6,
    )  # (num_snapshots+1, num_trajectories, num_coarse_grid)

    u_coarse = u_coarse_traj.permute(1, 0, 2)

    # --- Save ---
    print("Saving...")
    np.savez(
        "burgers.npz",
        u_ref    = u_ref.cpu().numpy(),
        dudt_ref = dudt_ref.cpu().numpy(),
        u_coarse = u_coarse.cpu().numpy(),
    )
    print(f"Done. Shape: {u_ref.shape}")


generate_dataset_torch()