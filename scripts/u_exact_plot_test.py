import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as anim
import torch

# Parameters
t_min, t_max = [0.0, 2.0]
x_min, y_min, x_max, y_max = [.0, 0.0, 1.0, 1.0]

# Time and grid setup
txy = torch.rand( (10000, 3) )
txy[:, 0] *= 0.0
txy[:, 1] = (x_max - x_min) * txy[:, 1] + x_min
txy[:, 2] = (y_max - y_min) * txy[:, 2] + y_min

alpha = 0.005
def u_exact(txy):
    t = txy[:, 0:1]
    x = txy[:, 1:2]
    y = txy[:, 2:3]

    u_3_4 = torch.sin(3 * torch.pi * x) * torch.sin(4 * torch.pi * y)
    u_8_6 = torch.sin(8 * torch.pi * x) * torch.sin(6 * torch.pi * y)
    u_2_3 = torch.sin(2 * torch.pi * x) * torch.sin(3 * torch.pi * y)
    a = alpha * torch.pi**2 * t
    c = 1
    
    return torch.exp(-25 * a) * u_3_4 + 0.5 * torch.exp(-100 * a) * u_8_6 + c * (1.0 - torch.exp(-5*t)) * u_2_3

fig, ax = plt.subplots()
#ax.autoscale(False)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.axis("equal")

levels = np.linspace(-1.0, 1.0, 81)
#levels = 81
triang = tri.Triangulation(txy[:, 1], txy[:, 2])
#txy[:, 0] = torch.ones(txy.shape[0]) * 0.5
tcf = ax.tricontourf(triang, u_exact(txy).squeeze(), levels=levels, extend='both', cmap='coolwarm')
fig.colorbar(tcf, ax=ax)

fps = 60
writer = anim.FFMpegWriter(
    fps=fps, 
    codec='libx264', 
    extra_args=[
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        "-threads","0"])
with writer.saving(fig, "u_exact_plot_test.mp4", dpi=150):
    for time in np.linspace(t_min, t_max, int((t_max - t_min) * fps) ):
        for coll in tcf.collections:
            coll.remove()
        txy[:, 0] = torch.ones(txy.shape[0]) * time
        ax.set_title(f"time = {time:10.2f}")
        u = u_exact(txy).squeeze()
        tcf = ax.tricontourf(triang, u, levels=levels, extend='both', cmap='coolwarm')
        writer.grab_frame()
        print(f"t = {time:10.2f}, u_min = {u.min():10.2f}, u_max = {u.max():10.2f}")