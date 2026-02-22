from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

# Добавление корневой директории проекта в sys.path чтобы появилась
# возможность импортировать модули из NN_pytorch_BVP
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
from NN_pytorch_BVP.pinn import sample_points_2D

def vector_field1(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    u = torch.sin(y)
    v = torch.sin(x)
    return torch.cat([u, v], dim=1)

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    bounds = [-2.0*torch.pi, -2.0*torch.pi, 2.0*torch.pi, 2.0*torch.pi]    # xmin, ymin, xmax, ymax
    #xy = sample_points_2D(bounds=bounds, n=100, scheme="uniform", device=device)
    xy = torch.cartesian_prod(
        torch.linspace(bounds[0], bounds[2], 20),
        torch.linspace(bounds[1], bounds[3], 20)
    )

    uv = vector_field1(xy)
    u = uv[:, 0]
    v = uv[:, 1]
    c = torch.sqrt( u**2 + v**2 )

    fig, ax = plt.subplots()
    q = ax.quiver(xy[:, 0], xy[:, 1], u, v, c, pivot="mid")
    ax.set(
        title="(sin(y), sin(x))", 
        xlabel="$x$", ylabel="$y$", 
        xlim=[bounds[0], bounds[2]], ylim=[bounds[1], bounds[3]],
        aspect="equal"
    )
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.colorbar(q, ax=ax)
    plt.show()