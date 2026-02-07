from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io
import matplotlib.pyplot as plt

def sample_points_2D(bounds: list[float], n: int, scheme: str, 
                     sobol_engine: torch.quasirandom.SobolEngine = None, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    x_min, y_min, x_max, y_max = bounds

    if scheme == "equal": # Равномерное распределение точек по прямоугольнику не реализовано
        raise ValueError(f"Equal scheme for generating points in a rectangle has not been implemented yet!")
    elif scheme == "uniform":
        xy = torch.rand(n, 2, device=device)
        xy[:, 0] = xy[:, 0] * (x_max - x_min) + x_min
        xy[:, 1] = xy[:, 1] * (y_max - y_min) + y_min 
        return xy
    elif scheme == "sobol":
        xy = sobol_engine.draw(n)
        xy[:, 0] = xy[:, 0] * (x_max - x_min) + x_min
        xy[:, 1] = xy[:, 1] * (y_max - y_min) + y_min 
        return xy

def u_exact(tx, dtype=torch.float32):
    data = scipy.io.loadmat(Path.cwd() / 'data' / 'allen_cahn_chatgpt_ported.mat')
    t = torch.tensor(data["tt"][0], dtype=dtype)
    x = torch.tensor(data["x"][0], dtype=dtype)
    u = torch.tensor(data['uu'], dtype=dtype)

    # Normalize t and x to [-1, 1]
    t_min, t_max = t.min(), t.max()
    x_min, x_max = x.min(), x.max()
    tx_normalized = torch.zeros_like(tx)
    tx_normalized[:, 0] = 2.0 * ( tx[:, 0] - t_min ) / (t_max - t_min) - 1.0
    tx_normalized[:, 1] = 2.0 * ( tx[:, 1] - x_min ) / (x_max - x_min) - 1.0

    n_points = tx.shape[0]
    u_grid = u.unsqueeze(0).unsqueeze(0)  # (1, 1, n_x, n_t)
    grid = tx_normalized.view(1, 1, n_points, 2)  # shape: (1, 1, n_points, 2)
    interpolated = F.grid_sample(
        u_grid, 
        grid, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=True
    )
    return interpolated.view(n_points, 1)

if __name__ == "__main__":
    t_min, t_max = (0.0, 1.0)
    x_min, x_max = (-1.0, 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 10000
    tx = sample_points_2D([t_min, x_min, t_max, x_max], n, scheme="uniform", device=device)
    u_exact_arr = u_exact(tx).reshape(-1)
    fig, ax = plt.subplots(1, 1)
    ax.set(
        title="Reference solutions", 
        xlabel="t", ylabel="x", 
        xlim=(t_min, t_max), 
        ylim=(x_min, x_max)
    )
    pc = ax.tripcolor(tx[:, 0], tx[:, 1], u_exact_arr, 
        shading="flat", 
        cmap="jet")
    fig.colorbar(pc, ax=ax)
    plt.show()