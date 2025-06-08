import matplotlib.pyplot as plt
import math
import torch

def my_f(bounds, n):
    # ------------ basic checks ---------------------------------------------
    if len(bounds) != 4:
        raise ValueError("`bounds` must have four numbers [x_min, y_min, x_max, y_max].")
    x_min, y_min, x_max, y_max = map(float, bounds)
    if not (x_max > x_min and y_max > y_min):
        raise ValueError("Need x_max > x_min and y_max > y_min.")
    if n <= 0:
        raise ValueError("`n` must be a positive integer.")

    # ------------ choose grid dimensions -----------------------------------
    width, height = x_max - x_min, y_max - y_min
    aspect      = width / height                        # rectangle shape
    cols        = math.ceil(math.sqrt(n * aspect))      # try to match aspect-ratio
    rows        = math.ceil(n / cols)                   # enough rows for n points

    # ------------ grid spacing (open rectangle) ----------------------------
    dx = width  / (cols + 1)     # +1 keeps every column at least one step away
    dy = height / (rows + 1)     #   from the respective border

    # ------------ generate points row-major until we reach n ---------------
    pts = []
    for r in range(rows):
        y = y_min + (r + 1) * dy
        for c in range(cols):
            x = x_min + (c + 1) * dx
            pts.append([x, y])
            if len(pts) == n:
                break
        if len(pts) == n:
            break

    return torch.tensor(pts)

if __name__ == "__main__":
    bounds = [-3.0, -2.0, 5.0, 4.0]
    n = 100

    points = my_f(bounds, n)

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0].cpu(), points[:, 1].cpu())     # convert to CPU if on GPU
    ax.add_patch(plt.Rectangle((bounds[0], bounds[1]),
                            bounds[2] - bounds[0],
                            bounds[3] - bounds[1],
                            fill=False, linestyle="--"))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f"{n} points inside rectangle {bounds}")
    plt.show()