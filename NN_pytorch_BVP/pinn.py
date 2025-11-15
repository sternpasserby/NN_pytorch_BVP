from pathlib import Path

import torch
import torch.nn as nn

def initialize_weights(model, scheme):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if scheme == 'naive':
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, mean=0.0, std=1.0)
            elif scheme == 'glorot_uniform':
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif scheme == 'glorot_normal':
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            else:
                raise ValueError(f"{scheme} is an unknown scheme for weights initialization")

def sample_points_1D(bounds: list[float], n: int, scheme: str, sobol_engine: torch.quasirandom.SobolEngine = None) -> torch.Tensor:
    a, b = bounds

    if scheme == "equal":
        return torch.linspace(a, b, n + 2)[1:-1].unsqueeze(1)
    elif scheme == "uniform":
        return (b - a) * torch.rand(n, 1) + a
    elif scheme == "sobol":
        if sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        points = sobol_engine.draw(n)
        return (b - a) * points + a
    else:
        raise ValueError(f"Unknown collocation points sampling scheme '{scheme}'.")

def sample_points_2D(bounds: list[float], n: int, scheme: str, 
                     sobol_engine: torch.quasirandom.SobolEngine = None, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    x_min, y_min, x_max, y_max = bounds

    if scheme == "equal": # Равномерное распределение точек по прямоугольнику не реализовано
        raise ValueError(f"Equal scheme for generating points in a rectangle has not been implemented yet!")
    elif scheme == "uniform":
        xy = torch.rand(n, 2, device=device)
        xy[:, 0] = xy[:, 0] * (x_max - x_min) + x_min
        xy[:, 1] = xy[:, 1] * (y_max - y_min) + y_min 
        return xy
    elif scheme == "sobol":
        xy = sobol_engine.draw(n)
        xy[:, 0] = xy[:, 0] * (x_max - x_min) + x_min
        xy[:, 1] = xy[:, 1] * (y_max - y_min) + y_min 
        return xy
    
def sample_points_3D(bounds: list[float], n: int, scheme: str, 
                     sobol_engine: torch.quasirandom.SobolEngine = None, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds

    if scheme == "equal": # Равномерное распределение точек по параллелепипеду не реализовано
        raise ValueError(f"Equal scheme for generating points in a parallelepiped has not been implemented yet!")
    elif scheme == "uniform":
        xyz = torch.rand(n, 3, device=device)
        xyz[:, 0] = xyz[:, 0] * (x_max - x_min) + x_min
        xyz[:, 1] = xyz[:, 1] * (y_max - y_min) + y_min
        xyz[:, 2] = xyz[:, 2] * (z_max - z_min) + z_min 
        return xyz
    elif scheme == "sobol":
        xyz = sobol_engine.draw(n)
        xyz[:, 0] = xyz[:, 0] * (x_max - x_min) + x_min
        xyz[:, 1] = xyz[:, 1] * (y_max - y_min) + y_min
        xyz[:, 2] = xyz[:, 2] * (z_max - z_min) + z_min 
        return xyz

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)            

def compute_grad_theta_norm(model):
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm(2).item()**2
    return grad_norm ** 0.5

class FourierFeatureEmbedding(nn.Module):
    """
    Explicit sinusoidal feature mapping layer.

    Args
    ----
    in_dim : int
        Dimensionality of the raw input (e.g. 2 for (x, y) pixels or 3 for (x, y, z) points).
    m : int
        Number of frequency bands (creates 2 * m * in_dim output channels).
    sigma : float
        A Gaussian σ; draws B~𝒩(0, σ⁻²) (like NeRF).
    """
    def __init__(self, in_dim: int, embed_dims: list[int], m: int, sigma: float):
        super().__init__()

        self.in_dim = in_dim
        self.embed_dims = embed_dims[:]
        self.m = m
        self.sigma = sigma

        self.register_buffer("B", torch.randn(m, len(embed_dims)) * sigma)

    @property
    def out_dim(self) -> int:
        """Useful when defining subsequent linear layers."""
        return 2 * self.m + self.in_dim - len(self.embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (..., in_dim) tensor in ℝ^{d}
        returns : (..., 2*num_frequencies*in_dim) tensor
        """
        # Ensure last dimension matches in_dim
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected input dimension {self.in_dim}, got {x.shape[-1]}")

        # F.linear does x @ B^T efficiently
        #phases = F.linear(x, self.B)

        x_to_embed = x[:, self.embed_dims]
        Bx = torch.matmul(x_to_embed, self.B.T)  # Shape (..., m)
        embedded = torch.cat((torch.cos(Bx), torch.sin(Bx)), dim=-1)
        other_idx = [i for i in range(x.shape[-1]) if i not in self.embed_dims]
        raw_rest = x[:, other_idx] if other_idx else None
        
        return embedded if raw_rest is None else torch.cat([raw_rest, embedded], dim=-1)

class PeriodicLayer(nn.Module):
    """
    Periodic feature mapping layer.

    This layer transforms selected input features into a periodic (Fourier-style) embedding.
    For each input feature `x_i` that is not in `keep_dims`, it produces the following features:

        [1, cos(ω_i*x_i), cos(2*ω_i*x_i), ..., cos(m*ω_i*x_i),
             sin(ω_i*x_i), sin(2*ω_i*x_i), ..., sin(m*ω_i*x_i)]

    Features specified in `keep_dims` are left unchanged and appended to the output.

    From the article "Sifan Wang, Shyam Sankaran: Respecting causality is all you need 
    for training physics-informed neural networks"

    Attributes:
        in_features (int): Number of input features.
        m (int): Number of harmonics (max multiplier for sine/cosine) for embedding.
        omega (list[float]): Frequency scaling for each input feature.
        keep_dims (list[int] or None): Indices of input features to leave unchanged.
        embed_dims (list[int]): Indices of input features to apply periodic embedding.
        _coeffs (torch.Tensor): Precomputed coefficients for embedding computation.
    """
    def __init__(self, in_features: int, m: int, omega: float | list[float], keep_dims: None | list[int] = None):
        """
        Initialize the PeriodicLayer.

        Args:
            in_features (int): Number of input features (dimensionality of input tensor).
            m (int): Number of harmonics to use for sine and cosine embeddings.
            omega (float or list of float): Frequency scaling. If a single float is provided, it is
                applied to all input features. If a list is provided, its length must equal `in_features`.
            keep_dims (list[int] or None, optional): Indices of input features that should
                be kept as-is (not transformed). Default is None (all features are transformed).
        """
        super().__init__()

        self.in_features = in_features
        self.m = m
        if isinstance(omega, (int, float)):
            self.omega = [omega] * in_features
        elif isinstance(omega, list):
            if len(omega) != in_features:
                raise ValueError("Length of omega list must equal in_features!")
            self.omega = omega[:]

        self.keep_dims = None if keep_dims is None else keep_dims[:]
        
        if keep_dims is None:
            embed_dims = list(range(in_features))
        else:
            embed_dims = [i for i in range(in_features) if i not in keep_dims]
            if len(embed_dims) == 0:
                raise ValueError("'keep_dims' list must not cover all the input features!")
        self.embed_dims = embed_dims

        self.register_buffer('_coeffs', torch.tensor([[(i + 1) * self.omega[j] for j in self.embed_dims] for i in range(self.m)]))

        self.out_features = len(embed_dims) * (1 + 2*m)
        if  isinstance(self.keep_dims, list):
            self.out_features += len(self.keep_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PeriodicLayer.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_features)`.

        Returns:
            torch.Tensor: Output tensor with shape `(batch_size, output_features)` where
                `output_features = len(embed_dims) * (2*m + 1) + len(keep_dims)`.
                Contains periodic embeddings for selected features and raw values for `keep_dims`.
        """
        ones_tensor = x.new_ones((x.shape[0], len(self.embed_dims)))
        x_embed = x[:, self.embed_dims]
        tmp = x_embed * self._coeffs
        result = torch.cat( (ones_tensor, torch.cos(tmp), torch.sin(tmp)), dim=1 )
        if self.keep_dims is None:
            return result
        else:
            return torch.cat((result, x[:, self.keep_dims]), dim=1)

# --- КЛАСС ПОЛНОСВЯЗНОЙ НЕЙРОННОЙ СЕТИ С FOURIER FEATURE EMBEDDING ---
class MultilayerPerceptronWithFFE(nn.Module):
    def __init__(self, layer_sizes, init_scheme, activation_fn=nn.Tanh(), use_FFE=True, FFE_embed_dims: list[int] = [], FFE_m=100, FFE_sigma=1.0):
        super().__init__()

        layer_sizes = layer_sizes[:]
        self.init_scheme = init_scheme
        self.activation_fn = activation_fn
        self.use_FFE = use_FFE 
        self.FFE_embed_dims = list(range(layer_sizes[0])) if len(FFE_embed_dims) == 0 else FFE_embed_dims[:]
        self.FFE_m = FFE_m
        self.FFE_sigma = FFE_sigma

        if use_FFE:
            layers = [FourierFeatureEmbedding(layer_sizes[0], self.FFE_embed_dims, FFE_m, FFE_sigma)]
            layer_sizes[0] = layers[0].out_dim
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
    
    @classmethod
    def save(cls, model, path: Path):
        """
        Saves model (both state_dict and additional parameters) to path
        """
        device = next(model.parameters()).device
        config = {
            "layer_sizes": model.layer_sizes,
            "init_scheme": model.init_scheme,
            "activation_fn": model.activation_fn,
            "use_FFE": model.use_FFE,
            "FFE_embed_dims": model.FFE_embed_dims[:],
            "FFE_m": model.FFE_m,
            "FFE_sigma": model.FFE_sigma,
            "device": str(device)
        }
        if model.use_FFE:
            config['layer_sizes'][0] = model.layers[0].in_dim
        torch.save({"config": config, "state_dict": model.state_dict()}, path)

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