"""
data_generation.py

Generates trajectories for the Burgers' equation closure model,
following the setup in Melchers et al. (2023).

"""

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


# Physical and discretisation parameters (from paper Appendix A.1)

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


def burgers_discretized(u, delta_x, nu=nu):
    """
    Discretize the Burgers' equation using Jameson's flux.

    Parameters
    u  : vector
    delta_x : grid spacing
    nu : viscosity

    Returns
    du/dt : vector time derivative
    """
    u_minus = np.roll(u,  1)   # u_{i-1}
    u_plus = np.roll(u, -1)   # u_{i+1}

    term1 = (nu / delta_x**2) * (u_minus - 2*u + u_plus)

    # flux at i+1/2 : f_{(i+1)/2}
    alpha_plus = 0.25 * np.abs(u + u_plus) - (1/12) * (u_plus - u)
    f_plus = (1/6) * (u**2 + u*u_plus + u_plus**2) - alpha_plus * (u_plus - u)

    # flux at i-1/2 : : f_{(i-1)/2}
    alpha_minus = 0.25 * np.abs(u_minus + u) - (1/12) * (u - u_minus)
    f_minus = (1/6) * (u_minus**2 + u_minus*u + u**2) - alpha_minus * (u - u_minus)

    term2 = -(f_plus - f_minus) / delta_x

    derivative = term1 + term2

    return derivative


def burgers_discretized_scipy(num_grid_points):
    """
    Wrapper for scipy solve_ivp requires function with signature (t, u).
    """
    delta_x = x_range / num_grid_points
    def rhs(t, u):
        return burgers_discretized(u, delta_x)
    return rhs


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


def downsample(u_fine):
    """
    Downsample from fine grid to coarse grid by averaging over blocks of 64.
    """
    factor = num_fine_grid // num_coarse_grid

    if u_fine.ndim == 1:
        return u_fine.reshape(-1, factor).mean(axis=1)
    else:
        # Shape: (nt, num_fine_grid)
        nt = u_fine.shape[0]
        return u_fine.reshape(nt, num_coarse_grid, factor).mean(axis=2)


def solve_grid(u0_fine, num_grid):
    """
    Solve Burgers' equation on the given grid from t=0 to t_final.
    Snapshots are saved at every snap_timestep interval.

    Returns
    -------
    u_fine : Shape (num_snapshots+1, num_grid)
    """
    derivative = burgers_discretized_scipy(num_grid)
    t_eval  = np.arange(0, num_snapshots + 1) * snap_timestep

    sol = solve_ivp(
        derivative,
        t_span=(0.0, t_final),
        y0=u0_fine,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-6,
    )

    return sol.y.T # Shape: (num_snapshots+1, nx_fine)

def compute_closure_targets(u_fine_traj):
    """
    Given a fine-grid trajectory, computes the following:
      - the downsampled reference solution u_ref
      - the downsampled reference derivative dudt_ref
    """

    u_ref    = downsample(u_fine_traj)
    dudt_ref = np.zeros_like(u_ref)

    Nt = u_fine_traj.shape[0]
    dx_fine = x_range / num_fine_grid

    for i in range(Nt):
        # Time derivative of the downsampled solution:
        # downsample the fine-grid RHS (consistent with the restriction operator)
        dudt_fine       = burgers_discretized(u_fine_traj[i], dx_fine)
        dudt_ref[i]     = downsample(dudt_fine)

    return u_ref, dudt_ref

def generate_dataset(seed=42):
    """
    Generate the full dataset of trajectories, matching the paper (96 training + 32 test = 128 total).
    Also generates trajectories from corresponding downsampled initial conditions.
    Saves everything into one .npz file.

    Each file contains:
    u_ref (num_trajectories, num_snapshots+1, num_coarse_grid) : fine grid solution downsampled to a coarse grid. By definition the
                                                                    most accurate low-fidelity approximation.
    dudt_ref (num_trajectories, num_snapshots+1, num_coarse_grid) : time derivative of u_ref
    u_coarse (num_trajectories, num_snapshots+1, num_coarse_grid) : coarse grid solution from downsampled initial conditions.
    """
    rng = np.random.default_rng(seed)

    all_u_ref    = []
    all_dudt_ref = []
    all_u_coarse = []

    for _ in tqdm(range(num_trajectories)):

        # 1. Generate fine-grid initial conditions and solve
        u0_fine      = make_initial_condition(num_fine_grid, rng)
        u_fine_traj  = solve_grid(u0_fine, num_fine_grid)

        # 2. Compute downsampled reference solution and targets
        u_ref, dudt_ref = compute_closure_targets(u_fine_traj)

        # 3. Solve coarse ODE from the downsampled initial conditions
        u0_coarse = downsample(u0_fine)
        u_coarse = solve_grid(u0_coarse, num_coarse_grid)

        all_u_ref.append(u_ref)
        all_dudt_ref.append(dudt_ref)
        all_u_coarse.append(u_coarse)

    # Stack into arrays: (num_trajectories, num_snapshots+1, num_coarse_grid)
    u_ref    = np.stack(all_u_ref)
    dudt_ref = np.stack(all_dudt_ref)
    u_coarse = np.stack(all_u_coarse)

    # Split into train / test
    def save(path):
        np.savez(path, u_ref=u_ref, dudt_ref=dudt_ref, u_coarse=u_coarse)
        print(f"Saved {path}")

    save("burgers.npz")

generate_dataset()
