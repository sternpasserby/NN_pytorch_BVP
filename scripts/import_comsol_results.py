from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as anim
from tqdm import trange
from scipy.interpolate import NearestNDInterpolator

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

    # Parameters
    x_min, x_max = -0.4, 0.94
    y_min, y_max = -0.1, 0.4
    t_min, t_max = 0.0, 10.0
    h = (y_max - y_min) / 2.0
    x_obs, y_obs, r_obs = x_min + 1.25*h, y_min + h*0.5, h/4.0    # x, y and r of an obstacle
    filepath_wo_ext = Path.cwd() / "data" / "navier-stokes_2d_incompressible_nonsteady_obstacle_shifted"
    device = torch.device("cpu")

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

    interpolator = NearestNDInterpolator(txy.detach().cpu(), uvp.detach().cpu())
    def uvp_ref(txy):
        uvp = torch.tensor(interpolator(txy.detach().cpu()), dtype=txy.dtype, device=txy.device)
        return uvp

    # Generate uniform mesh
    t = torch.linspace(t_min, t_max, 201, device=device)
    x = torch.linspace(x_min, x_max, 150, device=device)
    y = torch.linspace(y_min, y_max, 150, device=device)
    tmp = torch.cartesian_prod(torch.tensor([1.0]), x, y)
    mask = (tmp[:, 1] - x_obs)**2 + (tmp[:, 2] - y_obs)**2 > r_obs**2
    txy = tmp[mask]

    # Exclude obstacle from plots
    triang = tri.Triangulation(txy[:, 1], txy[:, 2])
    triangles = triang.triangles
    xmid = txy[:, 1][triangles].mean(axis=1)
    ymid = txy[:, 2][triangles].mean(axis=1)
    mask = (xmid - x_obs)**2 + (ymid - y_obs)**2 < r_obs**2
    triang.set_mask(mask)

    txy[:, 0] = txy[:, 0] * 0.0 + t[0]
    uvp = uvp_ref(txy)
    u = uvp[:, 0]
    v = uvp[:, 1]

    fig, ax = plt.subplots(1, 1, layout="compressed")
    pc1 = ax.tripcolor(triang, torch.sqrt( u.square() + v.square() ), 
        vmin=0.0, vmax=1.25, shading="gouraud", cmap="jet")
    ax.set(title=f"Velocity magnitude.\nt = {t[0]:.4f} s", xlabel="$x, m$", ylabel="$y, m$", aspect="equal")
    fig.colorbar(pc1, ax=ax).ax.set_title("m/s", pad=8)

    writer = anim.FFMpegWriter(
        fps=10, 
        codec='libx264', 
        extra_args=[
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", 
            "-preset", "ultrafast", 
            "-threads", "0"
        ]
    )
    pbar = trange(len(t), desc="Rendering import_comsol_results.mp4")
    with writer.saving(fig, Path.cwd() / "scripts" / "tmp" / "import_comsol_results.mp4", dpi=150):
        for iter in pbar:
            txy[:, 0] = txy[:, 0] * 0.0 + t[iter]
            uvp = uvp_ref(txy)
            u = uvp[:, 0]
            v = uvp[:, 1]

            pc1.set_array(torch.sqrt( u.square() + v.square() ))
            ax.set(title=f"Velocity magnitude.\nt = {t[iter]:.4f} s")

            writer.grab_frame()


    # # Plotting
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="compressed")
    # pc1 = ax1.tripcolor(triang, u, shading="gouraud", cmap="coolwarm")
    # ax1.set(title="u exact")
    # fig.colorbar(pc1, ax=ax1).ax.set_title("m/s", pad=8)
    # pc2 = ax2.tripcolor(triang, v, shading="gouraud", cmap="coolwarm")
    # ax2.set(title="v exact")
    # fig.colorbar(pc2, ax=ax2).ax.set_title("m/s", pad=8)
    # pc3 = ax3.tripcolor(triang, p, shading="gouraud", cmap="coolwarm")
    # ax3.set(title="p exact")
    # fig.colorbar(pc3, ax=ax3).ax.set_title("Pa", pad=8)
    # for ax in [ax1, ax2, ax3]:
    #     ax.set(xlabel="$x, m$", ylabel="$y, m$", aspect="equal")

    plt.show()