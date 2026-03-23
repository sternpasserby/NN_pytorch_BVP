import numpy as np
import matplotlib.pyplot as plt
import torch

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

if __name__ == "__main__":
    x = torch.linspace(-0.1, 0.2, 500)
    plt.plot(x, step_smoothed(x), label="step_smoothed(x)")
    plt.plot(x, step_smoothed(x, x0=0.05), label="step_smoothed(x - 0.05)")
    plt.plot(x, step_smoothed(x, transition_zone_size=0.2), label="step_smoothed(x, transition_zone_size=0.2)")
    plt.plot(x, step_smoothed(x, transition_zone_size=0.01), label="step_smoothed(x, transition_zone_size=0.01)")
    plt.plot(x, step_smoothed(x, min_value=-1.0, max_value=0.5), label="step_smoothed(x, min_value=-1.0, max_value=0.5)")
    plt.legend()
    plt.grid()
    plt.show()