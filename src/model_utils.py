#Imports
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from statsmodels.tsa.stattools import acf
from scipy.stats import kurtosis, gaussian_kde, rankdata, norm, genpareto, t
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import gamma
from sklearn.utils import resample
from types import SimpleNamespace
import numpy as np


def power_spherical_kernel(w, mu, kappa):
    # Assumes both w and mu are unit vectors of shape (d,)
    d = len(w)
    eta = (d - 1) / 2
    inner = (1 + np.dot(w, mu)) / 2  # between 0 and 1
    C = (4 * np.pi)**(-eta) * gamma(2 * eta + kappa) / gamma(eta + kappa)
    return C * (inner ** kappa)

def kde_power_spherical(w_query, w_data, kappa):
    # w_query: shape (N, d) -- query points on sphere
    # w_data: shape (n, d) -- observed angular samples
    N, d = w_query.shape
    n = w_data.shape[0]
    f_vals = np.zeros(N)
    for i in range(N):
        kernel_vals = [power_spherical_kernel(w_query[i], mu, kappa) for mu in w_data]
        f_vals[i] = np.mean(kernel_vals)
    return f_vals

def leave_one_out_nll(w_data, kappa, m=None, seed=42):
    """
    Approximate Leave-One-Out Negative Log-Likelihood for KDE with power spherical kernel.
    
    Args:
        w_data: ndarray (n, d), unit vectors on the sphere
        kappa: float, kernel concentration parameter
        m: int or None, number of points to subsample (for stochastic estimate)
        seed: int, random seed for reproducibility

    Returns:
        nll: float, average negative log-likelihood
    """
    np.random.seed(seed)
    n, d = w_data.shape

    if m is None or m >= n:
        idxs = np.arange(n)
    else:
        idxs = np.random.choice(n, size=m, replace=False)

    nlls = []
    for i in idxs:
        w_i = w_data[i]
        # Leave-one-out: remove w_i from data
        others = np.delete(w_data, i, axis=0)
        density = kde_power_spherical(w_i[None, :], others, kappa)[0]
        if density <= 0 or np.isnan(density):
            return np.inf
        nlls.append(-np.log(density))

    return np.mean(nlls)

def optimize_kappa(w_data, kappa_grid=None, m=1000, verbose = False):
    if kappa_grid is None:
        # Grid from 10^1 to 10^4
        kappa_grid = np.logspace(1, 4, 50)

    nll_values = []
    for kappa in kappa_grid:
        nll = leave_one_out_nll(w_data, kappa, m=m)
        if verbose:
            print(f"kappa = {kappa:.1f}, NLL = {nll:.4f}")
        nll_values.append(nll)

    best_idx = np.argmin(nll_values)
    best_kappa = kappa_grid[best_idx]
    if verbose:
        print(f"✅ Best kappa = {best_kappa:.2f} with NLL = {nll_values[best_idx]:.4f}")

    return best_kappa, kappa_grid, nll_values

def plot_kappa_optimization(kappa_grid, nll_values):
    plt.figure()
    plt.semilogx(kappa_grid, nll_values, marker='o')
    plt.xlabel("kappa")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Optimization of kappa (Power Spherical KDE)")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
     
def sample_power_spherical(w_data, n_samples, kappa, seed=None):
    """
    Sample unit vectors from power spherical KDE (approximate method).
    """
    if seed is not None:
        np.random.seed(seed)
    
    n, d = w_data.shape
    centers = w_data[np.random.choice(n, size=n_samples, replace=True)]  # (n_samples, d)

    noise = np.random.randn(n_samples, d)
    noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)  # unit noise

    # Interpolation between center and noise (approximate sampling)
    w_samples = (1 - (1 / kappa)) * centers + (1 / kappa) * noise
    w_samples = w_samples / np.linalg.norm(w_samples, axis=1, keepdims=True)
    return w_samples

def simulate_from_spar_deep(n_samples, model, u_model, w_ext, best_kappa, seed=None):
    """
    Simulate from SPAR model using power spherical KDE sampling.
    
    Args:
        n_samples: int
        model: neural net returning (sigma, xi)
        u_model: neural net returning u(w)
        w_ext: np.ndarray (n_ext, d), angular extremes
        best_kappa: float
        seed: int or None
    
    Returns:
        samples: np.ndarray (n_samples, d)
    """
    # 1. Sample angular directions W ~ KDE(w_ext)
    W_sim = sample_power_spherical(w_ext, n_samples, best_kappa, seed=seed)
    W_tensor = torch.tensor(W_sim, dtype=torch.float32)

    # 2. Predict GP parameters and threshold
    with torch.no_grad():
        sigma_sim, xi_sim = model(W_tensor)
        u_sim = u_model(W_tensor)

    sigma_sim = sigma_sim.numpy()
    xi_sim = xi_sim.numpy()
    u_sim = u_sim.numpy()

    # 3. Sample radial component
    R_sim = genpareto.rvs(c=xi_sim, scale=sigma_sim, size=n_samples) + u_sim

    # 4. Convert to cartesian
    samples = R_sim[:, None] * W_sim
    return samples
    

def fit_and_sample_gaussian_copula(resid_df, n_samples=10000, seed=42):
    """
    Fit a Gaussian copula on historical returns and simulate scenarios.
    
    Args:
        resid_df : pd.DataFrame (T x d) historical returns
        n_samples : int, number of scenarios to simulate
        seed : int, random seed
    
    Returns:
        simulated_df : pd.DataFrame, simulated returns (same scale as input)
    """
    np.random.seed(seed)
    
    # Step 1: Data setup
    X = resid_df.values
    n, d = X.shape
    tickers = resid_df.columns.tolist()
    
    # Step 2: Compute pseudo-observations (empirical CDF)
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    U = ranks / (n + 1)  # avoid 0/1 extremes
    
    # Step 3: Transform to Gaussian space
    Z = norm.ppf(U)  # inverse standard normal CDF
    Z = np.clip(Z, -10, 10)  # numerical stability
    
    # Step 4: Fit Gaussian copula = correlation matrix of Z
    corr_matrix = np.corrcoef(Z, rowvar=False)
    
    # Step 5: Simulate from multivariate normal with fitted correlation
    L = np.linalg.cholesky(corr_matrix)
    Z_sim = np.random.randn(n_samples, d) @ L.T
    
    # Step 6: Transform back to uniform via Φ
    U_sim = norm.cdf(Z_sim)
    
    # Step 7: Map U_sim to original data using empirical inverse CDF
    X_sim = np.zeros_like(U_sim)
    for j in range(d):
        sorted_vals = np.sort(X[:, j])
        idx = (U_sim[:, j] * (n - 1)).astype(int)
        idx = np.clip(idx, 0, n - 1)
        X_sim[:, j] = sorted_vals[idx]
    
    # Return DataFrame
    simulated_df = pd.DataFrame(X_sim, columns=tickers)
    return simulated_df
    
def fit_and_sample_t_copula(resid_df, nu=4, n_samples=10000, seed=42):
    
    """
    Fit a t-copula on historical returns and simulate scenarios.

    Args:
        resid_df : pd.DataFrame (T x d) historical returns
        nu : degrees of freedom of the t-copula
        n_samples : int, number of scenarios to simulate
        seed : int, random seed

    Returns:
        simulated_df : pd.DataFrame, simulated returns (same scale as input)
    """
    np.random.seed(seed)

    # Step 1: Data setup
    X = resid_df.values
    n, d = X.shape
    tickers = resid_df.columns.tolist()

    # Step 2: Compute pseudo-observations (empirical CDF)
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    U = ranks / (n + 1)  # avoid 0/1 extremes

    # Step 3: Transform to t space (inverse CDF of Student-t with df=nu)
    Z = t.ppf(U, df=nu)
    Z = np.clip(Z, -10, 10)  # numerical stability

    # Step 4: Fit correlation matrix from Z
    corr_matrix = np.corrcoef(Z, rowvar=False)

    # Step 5: Simulate from multivariate t distribution
    # Generate standard normal samples and scale by sqrt(W/nu)
    L = np.linalg.cholesky(corr_matrix)
    W = np.random.chisquare(df=nu, size=n_samples) / nu  # scaling variable
    Z_sim = np.random.randn(n_samples, d) @ L.T / np.sqrt(W)[:, None]

    # Step 6: Transform back to uniform via t CDF
    U_sim = t.cdf(Z_sim, df=nu)

    # Step 7: Map U_sim to original data using empirical inverse CDF
    X_sim = np.zeros_like(U_sim)
    for j in range(d):
        sorted_vals = np.sort(X[:, j])
        idx = (U_sim[:, j] * (n - 1)).astype(int)
        idx = np.clip(idx, 0, n - 1)
        X_sim[:, j] = sorted_vals[idx]

    simulated_df = pd.DataFrame(X_sim, columns=tickers)
    return simulated_df

def extract_spar_tail(df, u_model, mean_vec=None, std_vec=None, return_RW=True):
    """
    Extract tail points using SPAR's direction-dependent threshold u_model(W).
    
    Args:
        df : pd.DataFrame, original or simulated data
        u_model : trained SPAR u_model (PyTorch model)
        mean_vec, std_vec : optional normalization parameters (from SPAR training)
        return_RW : bool, whether to return (R_tail, W_tail)
    
    Returns:
        tail_df : DataFrame of tail points
        (optional) R_tail, W_tail : arrays
    """
    X = df.values
    n, d = X.shape

    # Step 1: Standardize
    if mean_vec is None:
        mean_vec = X.mean(axis=0)
    if std_vec is None:
        std_vec = X.std(axis=0)
    X_std = (X - mean_vec) / std_vec

    # Step 2: Compute R and W
    R = np.linalg.norm(X_std, axis=1)
    W = X_std / R[:, None]

    # Step 3: Predict thresholds with u_model
    W_tensor = torch.tensor(W, dtype=torch.float32)
    thresholds = u_model(W_tensor).detach().numpy().flatten()

    # Step 4: Filter tail points
    mask_tail = R > thresholds
    X_tail = X[mask_tail]
    X_bulk = X[~mask_tail]
    tail_df = pd.DataFrame(X_tail, columns=df.columns)
    bulk_df = pd.DataFrame(X_bulk, columns=df.columns)
    return tail_df, bulk_df    

def sample_equal_tail(tail, size):
    replace = size > len(tail)  
    indices = np.random.choice(len(tail), size=size, replace=replace)
    return tail.iloc[indices]

def resample_bulk(bulk, size):
    size = round(size)
    indices = np.random.choice(len(bulk), size=size, replace=True)
    return bulk.iloc[indices]

def rebuild_equal_dataset(tail, bulk, n_tail_target, factor=9):
    tail_sample = sample_equal_tail(tail, n_tail_target)
    bulk_sample = resample_bulk(bulk, n_tail_target * factor)
    mixed_dataset = pd.concat([bulk_sample, tail_sample])
    return mixed_dataset, tail_sample
