import math

import torch
import matplotlib.pyplot as plt

def sample_points_rectangle(n: int, xlims: list[float], ylims: list[float], 
        device: torch.device = torch.device("cpu")) -> torch.Tensor:
    x_min, x_max = xlims
    y_min, y_max = ylims

    xy = torch.rand(n, 2, device=device)
    xy[:, 0] = xy[:, 0] * (x_max - x_min) + x_min
    xy[:, 1] = xy[:, 1] * (y_max - y_min) + y_min 

    return xy

def sample_rectangle_minus_circle(n: int, xlims: list[float], ylims: list[float], 
        coords_c: list[float], r: float, device: torch.device = torch.device("cpu"),
        oversample_factor: float = 1.2) -> torch.Tensor:
    Lx = xlims[1] - xlims[0]
    Ly = ylims[1] - ylims[0]
    cx, cy = coords_c
    A_r = Lx * Ly
    A_c = torch.pi * r * r
    
    xy_total = []
    n_total = 0
    while n_total < n:
        N = math.ceil( oversample_factor * (n - n_total) * A_r / (A_r - A_c) )
        xy = sample_points_rectangle(N, xlims, ylims, device=device)
        mask = (xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2 > r**2 
        xy_total.append(xy[mask])
        n_total += len(xy_total[-1])

    return torch.cat(xy_total)[:n, :]

if __name__ == "__main__":
    device = torch.device("cuda")

    xlims = [-3.0, -1.0]    # [xmin, xmax]
    ylims = [-0.1, 0.4]    # [ymin, ymax]
    h = (ylims[1] - ylims[0]) / 2
    coords_c = [xlims[0] + 1.25*h, ylims[0] + h]
    r = 0.25*h

    n = 1000
    xy = sample_rectangle_minus_circle(n, xlims, ylims, coords_c, r, device=device)
    x = xy[:, 0].detach().cpu()
    y = xy[:, 1].detach().cpu()
    print(len(xy))

    w, h = plt.rcParams['figure.figsize']
    w *= 1.5; h *= 1.5
    fig, ax = plt.subplots(1, 1, figsize=(1*w, 1*h), constrained_layout=True)
    ax.scatter(x, y, s=1)
    ax.add_patch(plt.Circle(coords_c, r, color='black', fill=False, linewidth=2))
    ax.set(xlabel="$x$", ylabel="$y$", xlim=xlims, ylim=ylims,aspect="equal")
    plt.show()
