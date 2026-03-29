from pathlib import Path
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange

# Добавление корневой директории проекта в sys.path чтобы появилась
# возможность импортировать модули из NN_pytorch_BVP
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
from NN_pytorch_BVP.pinn import MultilayerPerceptronWithFFE

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

if __name__ == "__main__":
    x_min, x_max = -5.0, 6.5
    u_exact = lambda x: torch.sin(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers_list = [1, 16, 128, 256, 1]

    lr = 1e-3
    n_iters = 10000
    n_points = 100
    logging_freq = 100

    model = MultilayerPerceptronWithFFE(
        layer_sizes=layers_list,
        init_scheme="glorot_normal",
        activation_fn=nn.Tanh(),
        use_FFE=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logged_layers, logged_names = get_trainable_layers(model)
    n_log = len(range(0, n_iters, logging_freq))
    n_logged_layers = len(logged_layers)
    metrics = {
        "step": torch.arange(0, n_iters, logging_freq),
        "loss": torch.zeros(n_log),
        "per_layer_weight_ms": torch.zeros(n_log, n_logged_layers),
        "per_layer_bias_ms": torch.zeros(n_log, n_logged_layers),
        "per_layer_weight_grad_ms": torch.zeros(n_log, n_logged_layers),
        "per_layer_bias_grad_ms": torch.zeros(n_log, n_logged_layers),
    }
    pbar = trange(n_iters, desc="Training model")
    for iter in pbar:
        x = x_min + (x_max-x_min)*torch.rand(n_points, 1, device=device)
        u_model = model(x)

        loss = torch.mean( (u_model - u_exact(x))**2 )

        optimizer.zero_grad()
        loss.backward()

        if iter % logging_freq == 0:
            i = iter // logging_freq
            metrics["loss"][i] = loss.detach().item()
            log_per_layer_ms(i, logged_layers, 
                metrics["per_layer_weight_ms"], 
                metrics["per_layer_bias_ms"], 
                metrics["per_layer_weight_grad_ms"], 
                metrics["per_layer_bias_grad_ms"]
            )
            pbar.set_postfix({'loss': metrics["loss"][i].item()})
            
        optimizer.step()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)
    for j in range(n_logged_layers):
        ax1.plot(metrics["step"], metrics["per_layer_weight_ms"][:, j], label=logged_names[j])
        ax2.plot(metrics["step"], metrics["per_layer_bias_ms"][:, j], label=logged_names[j])
        ax3.semilogy(metrics["step"], metrics["per_layer_weight_grad_ms"][:, j], label=logged_names[j])
        ax4.semilogy(metrics["step"], metrics["per_layer_bias_grad_ms"][:, j], label=logged_names[j])
    ax1.set(title="Weight mean square by layer")
    ax2.set(title="Bias mean square by layer")
    ax3.set(title="Weight gradient mean square by layer")
    ax4.set(title="Bias gradient mean square by layer")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set(xlabel="grad step")
        ax.legend()
        ax.grid(True)
    plt.show()
