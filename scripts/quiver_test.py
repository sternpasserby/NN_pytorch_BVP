from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

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
    bounds = [0.0, 0.0, 4.0*torch.pi, 2.0*torch.pi]    # xmin, ymin, xmax, ymax

    # Построение осей
    fig, ax = plt.subplots()
    ax.set(
        title="(sin(y), sin(x))", 
        xlabel="$x$", ylabel="$y$", 
        xlim=[bounds[0], bounds[2]], ylim=[bounds[1], bounds[3]],
        aspect="equal"
    )

    # Построение contourf, отражающий модуль векторного поля в каждой точке пр-ва
    xy_bg = torch.cartesian_prod(
        torch.linspace(bounds[0], bounds[2], 100),
        torch.linspace(bounds[1], bounds[3], 100)
    )
    uv_bg = vector_field1(xy_bg)
    u_bg, v_bg = uv_bg[:, 0], uv_bg[:, 1]
    c_bg = torch.sqrt( u_bg**2 + v_bg**2 )
    cf = ax.contourf(
        xy_bg[:, 0].reshape(100, 100), xy_bg[:, 1].reshape(100, 100), c_bg.reshape(100, 100), 
        levels=150, 
        cmap="jet"
    )
    # Специальная настройка чтобы colorbar масштабировался вместе с осями
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad="2%")
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label(r"$\|\mathbf{F}(x,y)\|$")

    # Построение quiver со стрелками, отражающими направление векторного поля в каждой точке пр-ва
    nx = 40
    ny = 20
    xy = torch.cartesian_prod(
        torch.linspace(bounds[0], bounds[2], nx),
        torch.linspace(bounds[1], bounds[3], ny)
    )
    uv = vector_field1(xy)
    u, v = uv[:, 0], uv[:, 1]
    q = ax.quiver(
        xy[:, 0], xy[:, 1], 
        u, v, 
        pivot="mid",
        color="white",
        edgecolor="black", linewidth=0.6)
    
    plt.show()