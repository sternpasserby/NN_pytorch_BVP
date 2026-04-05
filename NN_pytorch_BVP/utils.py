import torch

# Для аккуратного масштабирования пределов оси Y во время рендера анимации обучения
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

# Вычисление градиента тензора при помощи torch.autograd. Пример: dy_dx = compute_grad(y, x)
def compute_grad(y, x, create_graph=True, retain_graph=None):
    if retain_graph is None:
        retain_graph = create_graph
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), 
        create_graph=create_graph, retain_graph=retain_graph)[0]

# Вычисляет проекции скоростей u и v из потоковой функции psi, т.к. u = dpsi/dy, v = -dpsi/dx
def get_uv_from_psi(txy, psi, create_graph=True):
    _, v, u = torch.split(compute_grad(psi, txy, create_graph=create_graph), 1, dim=1) 
    return u, -v

def step_smoothed(
    x: torch.Tensor, x0: float = 0.0, 
    transition_zone_size: float = 0.1, min_value: float = 0.0, max_value: float = 1.0
) -> torch.Tensor:
    """
    Compute a smooth approximation of a step function.

    The function transitions from ``min_value`` to ``max_value`` around ``x0``.
    Outside the transition zone, the output is constant. Inside the transition
    zone, a quintic polynomial is used to ensure a smooth interpolation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x0 : float, default=0.0
        Center of the transition.
    transition_zone_size : float, default=0.1
        Width of the transition zone centered at ``x0``. The function equals
        ``min_value`` for ``x <= x0 - transition_zone_size / 2`` and
        ``max_value`` for ``x >= x0 + transition_zone_size / 2``.
    min_value : float, default=0.0
        Output value below the transition zone.
    max_value : float, default=1.0
        Output value above the transition zone.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``x``, containing the smoothed step values.

    Notes
    -----
    The interpolation inside the transition zone is given by a fifth-degree
    polynomial, which provides a smooth transition with continuous derivatives.

    """
    d = transition_zone_size / 2.0
    xi = x - x0

    t = xi / d
    middle = 0.5 + (15.0/16.0)*t - (5.0/8.0)*t**3 + (3.0/16.0)*t**5

    h = torch.where(
        xi <= -d, 
        torch.zeros_like(xi),
        torch.where(xi >= d, torch.ones_like(xi), middle)
    )

    return min_value + (max_value - min_value) * h

def get_trainable_layers(model):
    """
    Collect submodules that directly own a trainable weight tensor.

    This function traverses ``model`` recursively using ``model.named_modules()``
    and returns all non-root submodules that have a non-``None`` ``weight``
    attribute. The traversal order is preserved, so the returned lists can be
    used to define a stable layer ordering for logging and plotting.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model whose submodules are to be inspected.

    Returns
    -------
    tuple[list[torch.nn.Module], list[str]]
        A pair ``(layers, names)`` where

        - ``layers`` is the list of submodules with a non-``None`` weight tensor.
        - ``names`` is the list of corresponding module names from
          ``model.named_modules()``.

        Both lists have the same length, and ``names[i]`` is the name of
        ``layers[i]``.

    Notes
    -----
    This function selects modules by checking for a ``weight`` attribute, so it
    typically includes layers such as ``nn.Linear`` and other weighted modules,
    while skipping activation layers and container modules that do not directly
    store parameters.
    """
    layers = []
    names = []

    for name, module in model.named_modules():
        if name == "":
            continue
        
        if hasattr(module, "weight") and module.weight is not None:
            layers.append(module)
            names.append(name)

    return layers, names

def mean_square(x: torch.Tensor) -> float:
    return torch.mean(x.detach() ** 2).item()

def log_per_layer_ms(    # ms = mean square
    log_idx: int, 
    logged_layers: list, 
    weight_ms_history: torch.Tensor,
    bias_ms_history: torch.Tensor, 
    weight_grad_ms_history: torch.Tensor, 
    bias_grad_ms_history: torch.Tensor
) -> None:
    """
    Log per-layer parameter and gradient mean squares into preallocated tensors.

    For each layer in ``logged_layers``, this function stores the mean square of
    the layer weight tensor, bias tensor, weight gradient, and bias gradient at
    row ``log_idx`` of the corresponding history tensors. If a layer has no bias,
    or if a gradient is unavailable, ``NaN`` is written to the appropriate entry.

    Parameters
    ----------
    log_idx : int
        Row index in the history tensors corresponding to the current logging step.
    logged_layers : list[torch.nn.Module]
        Ordered list of layers to monitor. Each layer is expected to have a
        ``weight`` attribute, and may optionally have a ``bias`` attribute.
    weight_ms_history : torch.Tensor
        Tensor of shape ``(n_log, n_layers)`` storing per-layer weight mean
        squares over training.
    bias_ms_history : torch.Tensor
        Tensor of shape ``(n_log, n_layers)`` storing per-layer bias mean
        squares over training.
    weight_grad_ms_history : torch.Tensor
        Tensor of shape ``(n_log, n_layers)`` storing per-layer weight-gradient
        mean squares over training.
    bias_grad_ms_history : torch.Tensor
        Tensor of shape ``(n_log, n_layers)`` storing per-layer bias-gradient
        mean squares over training.

    Returns
    -------
    None
        The input history tensors are modified in place.

    Notes
    -----
    This function should be called after ``loss.backward()`` so that parameter
    gradients are available, and before gradients are cleared by
    ``optimizer.zero_grad()``. The ordering of ``logged_layers`` defines the
    column ordering in all history tensors.
    """
    for j, layer in enumerate(logged_layers):
        # parameter mean squares
        weight_ms_history[log_idx, j] = mean_square(layer.weight)
        if hasattr(layer, "bias") and layer.bias is not None:
            bias_ms_history[log_idx, j] = mean_square(layer.bias)
        else:
            bias_ms_history[log_idx, j] = float("nan")

        # gradient mean squares
        if layer.weight.grad is not None:
            weight_grad_ms_history[log_idx, j] = mean_square(layer.weight.grad)
        else: 
            weight_grad_ms_history[log_idx, j] = float("nan")
        if hasattr(layer, "bias") and layer.bias is not None and layer.bias.grad is not None:
            bias_grad_ms_history[log_idx, j] = mean_square(layer.bias.grad)
        else:
            bias_grad_ms_history[log_idx, j] = float("nan")