from pathlib import Path
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as anim
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
    layers_list = [1, 16, 512, 256, 1]

    lr = 1e-3
    n_iters = 5000
    n_points = 100
    logging_freq = 100

    render_video = True
    video_fps = 10
    video_dpi = 100
    video_render_freq = 100    # render a frame once every N gradient descent steps

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

    ### ДЛЯ АНИМАЦИИ ПРОЦЕССА ОБУЧЕНИЯ
    if render_video:
        writer = anim.FFMpegWriter(
            fps=video_fps, 
            codec='libx264', 
            extra_args=['-pix_fmt', 'yuv420p', '-preset', 'ultrafast', "-threads", "0"]
        )
        w, h = plt.rcParams['figure.figsize']
        w *= 0.6; h *= 0.6

        fig, axes = plt.subplots(
            4, n_logged_layers, 
            figsize=(n_logged_layers*w, 4*h), squeeze=False, constrained_layout=False)

        row_labels = ["weights", "biases", "weights grad", "biases grad"]
        for i, label in enumerate(row_labels):
            axes[i, 0].set_ylabel(label, rotation=90, labelpad=20)
        for j, name in enumerate(logged_names):
            axes[0, j].set_title(name, pad=10)

        n_bins = 100
        st = np.empty_like(axes)
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                if i == 0 or i == 2:
                    data = logged_layers[j].weight
                elif i == 1 or i == 3:
                    data = logged_layers[j].bias
                counts, edges = torch.histogram(torch.rand_like(data).cpu(), bins=n_bins)
                st[i, j] = axes[i, j].stairs(counts.cpu(), edges.cpu(), fill=True, baseline=0, color='C0')

    pbar = trange(n_iters, desc="Training model")
    with writer.saving(fig, Path.cwd() / "scripts" / "tmp" / "layer_weights_extraction.mp4", dpi=video_dpi) if render_video else nullcontext():
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

            if render_video and iter % video_render_freq == 0:
                fig.suptitle(f"Grad step: {iter:6d}")

                for i in range(axes.shape[0]):
                    for j in range(axes.shape[1]):
                        if i == 0:
                            data = logged_layers[j].weight
                        elif i == 1:
                            data = logged_layers[j].bias
                        elif i == 2:
                            data = logged_layers[j].weight.grad
                        elif i == 3:
                            data = logged_layers[j].bias.grad
                        if data is None:
                            continue
                        counts, edges = torch.histogram(data.detach().cpu(), bins=n_bins)
                        st[i, j].set_data(values=counts.cpu(), edges=edges.cpu())
                        axes[i, j].relim()
                        axes[i, j].autoscale_view()

                writer.grab_frame()

            optimizer.step()

    # Plotting per-layer weights and biases mean squares
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