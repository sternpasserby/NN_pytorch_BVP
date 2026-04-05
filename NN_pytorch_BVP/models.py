from pathlib import Path

import torch
import torch.nn as nn

from NN_pytorch_BVP.layers import FourierFeatureEmbedding, PeriodicLayer

def initialize_weights(model: nn.Module, scheme: str) -> None:
    """
    Initialize the weights of all linear layers in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model whose ``nn.Linear`` layers will be initialized in place.
    scheme : str
        Weight initialization scheme to apply. Supported values are:

        - ``"naive"``: initialize weights and biases from a standard normal distribution
        - ``"glorot_uniform"``: apply Xavier/Glorot uniform initialization to weights
          and set biases to zero
        - ``"glorot_normal"``: apply Xavier/Glorot normal initialization to weights
          and set biases to zero

    Returns
    -------
    None
        This function modifies ``model`` in place and returns nothing.

    Raises
    ------
    ValueError
        If ``scheme`` is not one of the supported initialization schemes.

    Notes
    -----
    Only instances of ``nn.Linear`` are affected. All other module types are
    left unchanged.

    Examples
    --------
    >>> model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    >>> initialize_weights(model, "glorot_uniform")
    """
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if scheme == "naive":
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, mean=0.0, std=1.0)
            elif scheme == "glorot_uniform":
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif scheme == "glorot_normal":
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            else:
                raise ValueError(
                    f"{scheme} is an unknown scheme for weights initialization"
                )

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)            

def compute_grad_theta_norm(model: nn.Module) -> float:
    """
    Compute the L2 norm of all available parameter gradients in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model whose parameter gradients are used to compute the global
        gradient norm.

    Returns
    -------
    float
        Euclidean norm of all parameter gradients. Parameters whose
        ``grad`` attribute is ``None`` are ignored.

    Notes
    -----
    This function computes the global gradient norm as

    .. math::

        \\left( \\sum_i \\| g_i \\|_2^2 \\right)^{1/2}

    where :math:`g_i` is the gradient tensor of the ``i``-th parameter with a
    non-``None`` gradient.

    Examples
    --------
    >>> loss.backward()
    >>> grad_norm = compute_grad_theta_norm(model)
    >>> isinstance(grad_norm, float)
    True
    """
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm(2).item() ** 2
    return grad_norm ** 0.5

# --- КЛАСС ПОЛНОСВЯЗНОЙ НЕЙРОННОЙ СЕТИ С FOURIER FEATURE EMBEDDING ---
class MultilayerPerceptronWithFFE(nn.Module):
    def __init__(self, layer_sizes, init_scheme, activation_fn=nn.Tanh(), 
                 use_FFE=True, FFE_m=100, FFE_sigma=1.0, FFE_keep_dims: None | list[int] = None):
        super().__init__()

        layer_sizes = layer_sizes[:]
        self.init_scheme = init_scheme
        self.activation_fn = activation_fn
        self.use_FFE = use_FFE 
        self.FFE_m = FFE_m
        self.FFE_sigma = FFE_sigma
        self.FFE_keep_dims = None if FFE_keep_dims is None else FFE_keep_dims[:]

        if use_FFE:
            layers = [FourierFeatureEmbedding(layer_sizes[0], FFE_m, FFE_sigma, FFE_keep_dims)]
            layer_sizes[0] = layers[0].out_features
        else:
            layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_fn)
        self.layers = nn.Sequential(*layers)

        self._layer_sizes = layer_sizes[:]

        initialize_weights(self, init_scheme)

    @property
    def layer_sizes(self):
        return self._layer_sizes[:]

    def forward(self, x):
        return self.layers(x)
    
    def to_checkpoint(self) -> dict:
        """
        Generates a dictionary that stores model's config and model's state_dict. Can be used to
        save model to a external storage
        """
        device = next(self.parameters()).device
        config = {
            "layer_sizes": list(self.layer_sizes),
            "init_scheme": self.init_scheme,
            "activation_fn": self.activation_fn,
            "use_FFE": self.use_FFE,
            "FFE_m": self.FFE_m,
            "FFE_sigma": self.FFE_sigma,
            "FFE_keep_dims": self.FFE_keep_dims,
            "device": str(device)
        }
        if self.use_FFE:
            config['layer_sizes'][0] = self.layers[0].in_features
        return {"config": config, "state_dict": self.state_dict()}

    @classmethod
    def save(cls, model, path: Path):
        """
        Saves model (both state_dict and additional parameters) to path
        """
        torch.save(model.to_checkpoint(), path)

    @classmethod
    def load(cls, path: Path, device=None):
        """
        Loads model saved with cls.save() method. 
        If device=None, tries to load model to GPU. Otherwise, loads to CPU
        Currently, the method always loads to cuda:0
        """
        tmp = torch.load(path, weights_only=False)

        config = tmp.get("config")
        if config is None:
            raise KeyError("Checkpoint missing 'config'. Cannot reconstruct model.")
        
        config_device = config.get("device", None)
        if device is None:
            if config_device is not None and config_device.startswith("cuda") and torch.cuda.is_available():
                map_location = "cuda:0"
            else:
                map_location = "cpu"
            device = torch.device(map_location)
        else:
            map_location = device
        
        model = cls(**{k: v for k, v in config.items() if k != "device"})
        model.to(device)
        model.load_state_dict(tmp["state_dict"])
        model.eval()

        return model