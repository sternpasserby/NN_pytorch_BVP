from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as anim
from tqdm import tqdm

def _extract_times_from_header(header_lines: list[str]) -> np.ndarray:
    
    # Find the line with "...@ t=...". That line contains all the time moments
    line_with_times = None
    for line in header_lines:
        if "@ t=" in line:
            line_with_times = line
            break
    if line_with_times is None:
        raise ValueError("Could not find a header line containing '@ t='.")

    # Parse the line_with_times in order to obtain raw time moments array
    parts = line_with_times.split("@ t=")
    raw_times = []
    for part in parts[1:]:    # parts[0] is the text before the first time stamp
        raw_times.append(float(part.strip().split()[0]))
    if len(raw_times) % 3 != 0:
        raise ValueError(
            f"Expected number of time entries to be divisible by 3, got {len(raw_times)}."
        )
    raw_times = np.asarray(raw_times, dtype=np.float64)

    # optional consistency check
    if not (
        np.allclose(raw_times[0::3], raw_times[1::3]) and
        np.allclose(raw_times[0::3], raw_times[2::3])
    ):
        raise ValueError("Header does not appear to contain (u,v,p) triplets per time step.")

    return raw_times[::3]    # u,v,p repeat the same time, so keep one from each triplet

def parse_comsol_txt(
    filepath: Path, 
    dtype: torch.dtype = torch.float32, 
    device: torch.device = torch.device("cpu")
) -> dict[str, torch.Tensor]:
    
    # Get time moments array
    header_lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("%"):
                header_lines.append(line.rstrip("\n"))
            else:
                break
    t = _extract_times_from_header(header_lines)

    # Get u, v and p for each time moment
    data = np.genfromtxt(filepath, comments="%")
    xy = data[:, 0:2]
    u = data[:, [2 + 3*i for i in range(len(t))]]
    v = data[:, [3 + 3*i for i in range(len(t))]]
    p = data[:, [4 + 3*i for i in range(len(t))]]

    to_pytorch = lambda x: torch.tensor(x, dtype=dtype, device=device)

    return {
        "xy": to_pytorch(xy), 
        "t": to_pytorch(t), 
        "u": to_pytorch(u), 
        "v": to_pytorch(v), 
        "p": to_pytorch(p)
    }

if __name__ == "__main__":

    # Parameters
    x_min, x_max = -0.4, 0.94
    y_min, y_max = -0.1, 0.4
    t_min, t_max = 0.0, 1.0
    h = (y_max - y_min) / 2.0
    x_obs, y_obs, r_obs = x_min + 1.25*h, y_min + h, h/4.0    # x, y and r of an obstacle
    comsol_solution_filepath = Path.cwd() / "data" / "navier-stokes_2d_incompressible_nonsteady_obstacle.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Exporting and importing Comsol solution
    comsol_solution_dict = parse_comsol_txt(comsol_solution_filepath, device=device)
    torch.save(comsol_solution_dict, comsol_solution_filepath.with_suffix(".pt"))
    comsol_solution_dict = torch.load(comsol_solution_filepath.with_suffix(".pt"), map_location=device)

    # Unpacking Comsol solution
    xy_comsol = comsol_solution_dict["xy"].cpu()
    t_comsol = comsol_solution_dict["t"].cpu()
    u_comsol = comsol_solution_dict["u"].cpu()
    v_comsol = comsol_solution_dict["v"].cpu()
    p_comsol = comsol_solution_dict["p"].cpu()

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

    # # Generate uniform mesh
    # t = torch.linspace(t_min, t_max, 201, device=device)
    # x = torch.linspace(x_min, x_max, 150, device=device)
    # y = torch.linspace(y_min, y_max, 150, device=device)
    # tmp = torch.cartesian_prod(torch.tensor([1.0]), x, y)
    # mask = (tmp[:, 1] - x_obs)**2 + (tmp[:, 2] - y_obs)**2 > r_obs**2
    # txy = tmp[mask]

    # Exclude obstacle from plots
    triang = tri.Triangulation(xy_comsol[:, 0], xy_comsol[:, 1])
    triangles = triang.triangles
    xmid = xy_comsol[:, 0][triangles].mean(axis=1)
    ymid = xy_comsol[:, 1][triangles].mean(axis=1)
    mask = (xmid - x_obs)**2 + (ymid - y_obs)**2 < r_obs**2
    triang.set_mask(mask)

    # Prepare figure and axes
    w, h = plt.rcParams['figure.figsize']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3*w, 1*h), layout="compressed")
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
    for ax in [ax1, ax2, ax3]:
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
    pbar = tqdm(times, desc="Rendering import_comsol_results.mp4")
    video_path = Path.cwd() / "tmp" / "import_comsol_results"
    video_path.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, video_path / "import_comsol_results.mp4", dpi=150):
        for t in pbar:
            u, v, p = uvp_ref(t)

            pc1.set_array(u)
            pc2.set_array(v)
            pc3.set_array(p)
            pc3.set_clim(p.min(), p.max())

            fig.suptitle(f"time = {t:10.2f}")

            writer.grab_frame()