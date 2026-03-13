from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Добавление корневой директории проекта в sys.path чтобы появилась
# возможность импортировать модули из NN_pytorch_BVP
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

def load_data(
    filepath: Path, 
    dtype: torch.dtype = torch.float32, 
    device: torch.device = torch.device("cpu")
) -> tuple["torch.Tensor", "torch.Tensor"]:
    data = np.genfromtxt(filepath, comments="%")

    txy = torch.tensor(data[:, [2, 0, 1]], dtype=dtype, device=device)
    uvp = torch.tensor(data[:, [3, 4, 5]], dtype=dtype, device=device)

    return txy, uvp

def load_pt_data(
    filepath: Path,
    device: torch.device = torch.device("cpu"),
) -> tuple["torch.Tensor", "torch.Tensor"]:
    data = torch.load(filepath, map_location=device)
    return data["txy"], data["uvp"]

if __name__ == "__main__":
    filepath_wo_ext = Path.cwd() / "data" / "navier-stokes_2d_incompressible_nonsteady_obstacle"
    x_obs, y_obs, r_obs = -0.0875, 0.15, 0.0625    # x, y and r of an obstacle
    t0 = 0.10325    # time moment to plot
    device = torch.device("cuda")

    # Importing and saving txy and uvp
    filepath_pt = filepath_wo_ext.with_suffix(".pt")
    filepath_txt = filepath_wo_ext.with_suffix(".txt")
    if not filepath_pt.is_file():    # if .pt does not exist: import txt and save pt
        print(f"File {str(filepath_pt)} does not exists. I am going to create it.")
        print(f"Importing {str(filepath_txt)}...", end="")
        txy, uvp = load_data(filepath_txt, device=device)
        print("Done")
        print(f"Saving to {str(filepath_pt)}...", end="")
        torch.save({"txy": txy, "uvp": uvp}, filepath_wo_ext.with_suffix(".pt"))
        print("Done")
    else:    # load pt
         txy, uvp = load_pt_data(filepath_pt, device=device)

    mask = torch.isclose(txy[:, 0], torch.tensor(t0, dtype=txy.dtype, device=txy.device), atol=1e-8)
    x = txy[mask, 1].detach().cpu()
    y = txy[mask, 2].detach().cpu()
    u = uvp[mask, 0].detach().cpu()
    v = uvp[mask, 1].detach().cpu()
    p = uvp[mask, 2].detach().cpu()

    # Exclude obstacle from plots
    triang = tri.Triangulation(x, y)
    triangles = triang.triangles
    xmid = x[triangles].mean(axis=1)
    ymid = y[triangles].mean(axis=1)
    mask = (xmid - x_obs)**2 + (ymid - y_obs)**2 < r_obs**2
    triang.set_mask(mask)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="compressed")
    pc1 = ax1.tripcolor(triang, u, shading="gouraud", cmap="coolwarm")
    ax1.set(title="u exact")
    fig.colorbar(pc1, ax=ax1).ax.set_title("m/s", pad=8)
    pc2 = ax2.tripcolor(triang, v, shading="gouraud", cmap="coolwarm")
    ax2.set(title="v exact")
    fig.colorbar(pc2, ax=ax2).ax.set_title("m/s", pad=8)
    pc3 = ax3.tripcolor(triang, p, shading="gouraud", cmap="coolwarm")
    ax3.set(title="p exact")
    fig.colorbar(pc3, ax=ax3).ax.set_title("Pa", pad=8)
    for ax in [ax1, ax2, ax3]:
        ax.set(xlabel="$x, m$", ylabel="$y, m$", aspect="equal")

    plt.show()