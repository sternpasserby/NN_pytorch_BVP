import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as anim
from scipy.special import jn_zeros, j0, j1
import torch

# Parameters
alpha = 1.0
R = 5
t_min, t_max = [0.0, 10.0]
x_min, y_min, x_max, y_max = [-R, -R, R, R]

# Time and grid setup
t_r_theta = torch.rand( (10000, 3) )
t_r_theta[:, 0] *= t_max
t_r_theta[:, 1] *= R
t_r_theta[:, 2] *= 2.0 * np.pi

txy = torch.empty_like(t_r_theta)
txy[:, 1] = t_r_theta[:, 1] * torch.cos(t_r_theta[:, 2])
txy[:, 2] = t_r_theta[:, 1] * torch.sin(t_r_theta[:, 2])

n = 10
mu_n = jn_zeros(0, n)
def u_exact(t, r, theta):
    tmp = torch.zeros_like(r)
    for i in range(n):
        tmp += j0(mu_n[i] * r * 0.2) * torch.exp( -(mu_n[i] * 0.2)**2 * t ) / ( -mu_n[i] * j1(mu_n[i]) )
    return tmp * 16 + 8

fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.axis("equal")

levels = np.linspace(0, 10, 81)
triang = tri.Triangulation(txy[:, 1], txy[:, 2])
tcf = ax.tricontourf(triang, u_exact(t_r_theta[:, 0] * 0.0, t_r_theta[:, 1], t_r_theta[:, 2]), levels=levels, extend='both', cmap='coolwarm')
fig.colorbar(tcf)

writer = anim.FFMpegWriter(
    fps=30, 
    codec='libx264', 
    extra_args=[
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        "-threads","0"])
with writer.saving(fig, "tmp.mp4", dpi=150):
    for time in np.linspace(t_min, t_max, int((t_max - t_min) * 30) ):
        for coll in tcf.collections:
            coll.remove()
        txy[:, 0] = torch.ones(txy.shape[0]) * time
        ax.set_title(f"time = {time:10.2f}")
        u = u_exact(t_r_theta[:, 0] * 0.0 + time, t_r_theta[:, 1], t_r_theta[:, 2])
        tcf = ax.tricontourf(triang, u, levels=levels, extend='both', cmap='coolwarm')
        writer.grab_frame()
        print(f"t = {time:10.2f}, u_min = {u.min():10.2f}, u_max = {u.max():10.2f}")