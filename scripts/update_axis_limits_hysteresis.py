import matplotlib.pyplot as plt
import matplotlib.animation as anim
import torch

def _relax_lim(
    ax: object,
    lims: tuple[float, float] | list[float],
    which: str,
    pad: float = 0.05,
    alpha_shrink: float = 0.08,
) -> None:
    """
    Update one axis limit range with hysteresis-like behavior.

    The target limits are first padded by a fraction of their span. The current
    axis limits are then adjusted so that expansion happens immediately, while
    shrinking toward tighter limits happens gradually.

    Parameters
    ----------
    ax
        Matplotlib axes object whose limits will be updated.
    lims
        New unpadded target limits as ``(min, max)``.
    which
        Axis selector: ``"x"`` for x-axis or ``"y"`` for y-axis.
    pad
        Fractional padding added on both sides of the target interval.
        For example, ``pad=0.05`` adds 5% of the target span below the minimum
        and above the maximum.
    alpha_shrink
        Smoothing factor used when shrinking limits toward the target interval.
        Must typically lie in ``[0, 1]``:
        - ``0`` means never shrink,
        - ``1`` means shrink immediately,
        - intermediate values produce gradual shrinkage.

    Raises
    ------
    ValueError
        If ``which`` is not ``"x"`` or ``"y"``.

    Notes
    -----
    The update is asymmetric:
    - if the new padded interval exceeds the current limits, the limits expand
      immediately to avoid clipping;
    - if the new padded interval is contained within the current limits, the
      limits move gradually toward it.

    This is useful for animated plots where ordinary autoscaling causes visible
    jitter or "breathing".
    """
    tmin, tmax = lims
    if tmin == tmax:
        tmin -= 1.0
        tmax += 1.0
    
    tr = tmax - tmin
    tmin_t = tmin - pad * tr
    tmax_t = tmax + pad * tr

    if which == "x":
        cur_tmin, cur_tmax = ax.get_xlim()
    elif which == "y":
        cur_tmin, cur_tmax = ax.get_ylim()
    else:
        raise ValueError("which must be 'x' or 'y'")
    
    # Expand immediately if needed
    if tmin_t < cur_tmin:
        cur_tmin = tmin_t
    else:  # shrink slowly
        cur_tmin = (1 - alpha_shrink) * cur_tmin + alpha_shrink * tmin_t
    if tmax_t > cur_tmax:
        cur_tmax = tmax_t
    else:
        cur_tmax = (1 - alpha_shrink) * cur_tmax + alpha_shrink * tmax_t

    if which == "x":
        ax.set_xlim(cur_tmin, cur_tmax)
    elif which == "y":
        ax.set_ylim(cur_tmin, cur_tmax)
    else:
        raise ValueError("which must be 'x' or 'y'")

def update_axis_limits_hysteresis(
    ax: object,
    xlims_new: tuple[float, float] | list[float] | None = None,
    ylims_new: tuple[float, float] | list[float] | None = None,
    pad: float = 0.05,
    alpha_shrink: float = 0.08,
) -> None:
    """
    Update x and/or y axis limits using hysteresis-like smoothing.

    This function applies asymmetric limit updates to the given axes:
    expansion is immediate when new data would exceed the current view, while
    shrinking toward tighter limits is performed gradually. This helps reduce
    visual jitter in animated or repeatedly updated plots.

    Parameters
    ----------
    ax
        Matplotlib axes object whose limits will be updated.
    xlims_new
        New target x-axis limits as ``(min, max)``. If ``None``, the x-axis is
        left unchanged.
    ylims_new
        New target y-axis limits as ``(min, max)``. If ``None``, the y-axis is
        left unchanged.
    pad
        Fractional padding added to each provided target interval before
        updating the displayed limits.
    alpha_shrink
        Smoothing factor used only when shrinking limits. See `_relax_lim`
        for details.

    Notes
    -----
    This function is intended for dynamic plotting, where raw autoscaling can
    make the axes appear unstable. It delegates the actual one-dimensional
    update logic to `_relax_lim`.
    """
    if xlims_new is not None:
        _relax_lim(ax, xlims_new, which="x", pad=pad, alpha_shrink=alpha_shrink)
    if ylims_new is not None:
        _relax_lim(ax, ylims_new, which="y", pad=pad, alpha_shrink=alpha_shrink)
    
if __name__ == "__main__":
    n_points = 500
    n_bins = 50
    mu_min, mu_max = -1.0, 1.0
    sigma_min, sigma_max = 1.0, 7.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(n_points, device=device)
    counts, bins = torch.histogram(x.detach().cpu(), bins=n_bins)

    fig, ax = plt.subplots()
    st = ax.stairs(counts.cpu(), bins.cpu(), fill=True, baseline=0, color='C0')

    def update(frame):
        mu = mu_min + (mu_max - mu_min) * torch.rand(1)
        sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(1)
        x = mu + sigma*torch.randn(n_points, device=device)

        counts, bins = torch.histogram(x.detach().cpu(), bins=n_bins)
        st.set_data(values=counts.cpu(), edges=bins.cpu())
        xlims_new = [bins.min().item(), bins.max().item()]
        ylims_new = [counts.min().item(), counts.max().item()]
        update_axis_limits_hysteresis(ax, xlims_new, ylims_new)

        return (st,)

    ani = anim.FuncAnimation(
        fig,
        update,
        frames=100,
        interval=100,   # milliseconds between frames
        blit=False,
        repeat=False,
    )

    plt.show()