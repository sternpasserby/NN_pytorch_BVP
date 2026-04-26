from pathlib import Path
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as anim
from tqdm import tqdm

# Добавление корневой директории проекта в sys.path чтобы появилась
# возможность импортировать модули из NN_pytorch_BVP
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
from NN_pytorch_BVP.models import MultilayerPerceptronWithFFE
from NN_pytorch_BVP.utils import get_uv_from_psi

def model_txy_to_uvp(model, txy):
    out = model(txy)
    psi, p = out[:, 0:1], out[:, 1]
    u, v = get_uv_from_psi(txy, psi, create_graph=False)
    return u.squeeze(), v.squeeze(), p

if __name__ == "__main__":

    # Parameters
    x_min, x_max = -0.4, 0.94
    y_min, y_max = -0.1, 0.4
    t_min, t_max = 0.0, 1.0
    h = (y_max - y_min) / 2.0
    x_obs, y_obs, r_obs = x_min + 1.25*h, y_min + h, h/4.0    # x, y and r of an obstacle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    comsol_solution_filepath = Path.cwd() / "data" / "navier-stokes_2d_incompressible_nonsteady_obstacle.txt"
    model_results_filepath = (
        Path.cwd() / 
        "runs" / 
        "05_03_navier-stokes_2d_incompressible_nonsteady_obstacle" / 
        "FFE256+CosineAnnealingLR_lr1e-4+bigger_network+GELU+curriculum"
    )
    model_filepath = model_results_filepath / "model" / "model_epoch=25000_batch=20.pth"
    config_filepath = model_results_filepath / "config.json"

    comsol_solution_dict = torch.load(comsol_solution_filepath.with_suffix(".pt"), map_location=device)
    model = MultilayerPerceptronWithFFE.load(model_filepath).to(device)
    with open(config_filepath, "r", encoding="utf-8") as f:
        config = json.load(f)
    L_wave = config["L_wave"]
    T_wave = config["T_wave"]
    U_wave = config["U_wave"]
    P_wave = config["P_wave"]

    # Unpacking Comsol solution
    xy_comsol = comsol_solution_dict["xy"].cpu()
    t_comsol = comsol_solution_dict["t"].cpu()
    u_comsol = comsol_solution_dict["u"].cpu()
    v_comsol = comsol_solution_dict["v"].cpu()
    p_comsol = comsol_solution_dict["p"].cpu()
  
    ones_arr = torch.ones_like(xy_comsol)[:, 0:1].detach()
    txy = torch.cat( (ones_arr, xy_comsol/L_wave), dim=1 ).detach().to(device).requires_grad_(True)
    ones_arr = ones_arr.squeeze().to(device).requires_grad_(True)

    # # Creating interpolator
    def uvp_ref(time_moment: float):
        if time_moment <= t_comsol[0]:
            return u_comsol[:, 0], v_comsol[:, 0], p_comsol[:, 0]
        if time_moment >= t_comsol[-1]:
            return u_comsol[:, -1], v_comsol[:, -1], p_comsol[:, -1]

        idx = torch.searchsorted(t_comsol, time_moment, right=True) - 1
        alpha = (time_moment - t_comsol[idx]) / (t_comsol[idx + 1] - t_comsol[idx])

        u = (1.0 - alpha)*u_comsol[:, idx] + alpha*u_comsol[:, idx+1]
        v = (1.0 - alpha)*v_comsol[:, idx] + alpha*v_comsol[:, idx+1]
        p = (1.0 - alpha)*p_comsol[:, idx] + alpha*p_comsol[:, idx+1]

        return u, v, p

    # Exclude obstacle from plots
    triang = tri.Triangulation(xy_comsol[:, 0], xy_comsol[:, 1])
    triangles = triang.triangles
    xmid = xy_comsol[:, 0][triangles].mean(axis=1)
    ymid = xy_comsol[:, 1][triangles].mean(axis=1)
    mask = (xmid - x_obs)**2 + (ymid - y_obs)**2 < r_obs**2
    triang.set_mask(mask)

    # Prepare figure and axes
    w, h = plt.rcParams['figure.figsize']
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(3*w, 2*h), layout="compressed")
    placeholder_tensor = torch.zeros_like(u_comsol[:, 0])
    pc1 = ax1.tripcolor(triang, placeholder_tensor, 
        vmin=u_comsol.min(), vmax=u_comsol.max(), shading="gouraud", cmap="jet")
    ax1.set(title="u")
    fig.colorbar(pc1, ax=ax1).ax.set_title("m/s", pad=8)
    pc2 = ax2.tripcolor(triang, placeholder_tensor, 
        vmin=v_comsol.min(), vmax=v_comsol.max(), shading="gouraud", cmap="jet")
    ax2.set(title="v")
    fig.colorbar(pc2, ax=ax2).ax.set_title("m/s", pad=8)
    pc3 = ax3.tripcolor(triang, placeholder_tensor, 
        vmin=p_comsol.min(), vmax=p_comsol.max(), shading="gouraud", cmap="binary")
    ax3.set(title="p")
    fig.colorbar(pc3, ax=ax3).ax.set_title("Pa", pad=8)
    pc4 = ax4.tripcolor(triang, placeholder_tensor, 
        vmin=u_comsol.min(), vmax=u_comsol.max(), shading="gouraud", cmap="jet")
    ax4.set(title="u")
    fig.colorbar(pc4, ax=ax4).ax.set_title("m/s", pad=8)
    pc5 = ax5.tripcolor(triang, placeholder_tensor, 
        vmin=v_comsol.min(), vmax=v_comsol.max(), shading="gouraud", cmap="jet")
    ax5.set(title="v")
    fig.colorbar(pc5, ax=ax5).ax.set_title("m/s", pad=8)
    pc6 = ax6.tripcolor(triang, placeholder_tensor, 
        vmin=p_comsol.min(), vmax=p_comsol.max(), shading="gouraud", cmap="binary")
    ax6.set(title="p")
    fig.colorbar(pc6, ax=ax6).ax.set_title("Pa", pad=8)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set(xlabel="x, m", ylabel="y, m", aspect="equal", xlim=[x_min, x_max], ylim=[y_min, y_max])

    writer = anim.FFMpegWriter(
        fps=10,    # 0.1 seconds per frame
        codec='libx264', 
        extra_args=[
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", 
            "-preset", "ultrafast", 
            "-threads", "0"
        ]
    )
    times = t_comsol.cpu().numpy()
    pbar = tqdm(times, desc="Rendering comsol_vs_neural_model.mp4")
    video_path = Path.cwd() / "tmp" / "render_comsol_vs_neural_model"
    video_path.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, video_path / "comsol_vs_neural_model.mp4", dpi=150):
        for t in pbar:
            u_ref, v_ref, p_ref = uvp_ref(t)

            with torch.no_grad():
                txy[:, 0] = ones_arr * t / T_wave

            u_model, v_model, p_model = model_txy_to_uvp(model, txy)
            u_model = u_model.detach().cpu() * U_wave
            v_model = v_model.detach().cpu() * U_wave
            p_model = p_model.detach().cpu() * P_wave

            pc1.set_array(u_ref)
            pc2.set_array(v_ref)
            pc3.set_array(p_ref)
            pc3.set_clim(p_ref.min(), p_ref.max())
            pc4.set_array(u_model)
            pc5.set_array(v_model)
            pc6.set_array(p_model)
            pc6.set_clim(p_model.min(), p_model.max())

            fig.suptitle(f"time = {t:10.2f}")

            writer.grab_frame()