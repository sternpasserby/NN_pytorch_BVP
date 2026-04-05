import torch
import torch.nn as nn

class FourierFeatureEmbedding(nn.Module):
    """
    Fourier feature positional encoding for selected input dimensions.

    This layer projects a subset of the input features into an `m`-dimensional
    Gaussian random feature space and returns their cosine and sine embeddings.
    Optionally, some of the original input dimensions can be preserved and
    concatenated to the output.

    Args:
        in_features (int):
            Total number of input features.
        m (int):
            Number of random Fourier features per input dimension.
            The output from the embedding is of size `2 * m` for the
            embedded dimensions.
        sigma (float):
            Standard deviation of the Gaussian used to initialize the
            projection matrix `B` ∈ R^{len(embed_dims) × m}.
        keep_dims (list[int] or None):
            Indices of input dimensions to be passed through unchanged.
            If `None`, all input dimensions are Fourier-embedded.
            If provided, all *other* dimensions are embedded.

    Attributes:
        embed_dims (list[int]):
            Indices of input features that will be Fourier-embedded.
        B (Tensor):
            Random projection matrix with shape `(len(embed_dims), m)`.
        out_features (int):
            Output dimensionality after embedding and optional passthrough.

    Shape:
        - Input:  `(N, in_features)`
        - Output: `(N, out_features)`
          where `out_features = 2*m + len(keep_dims or [])`

    Raises:
        ValueError:
            If `keep_dims` covers all input dimensions.

    """
    def __init__(self, in_features: int, m: int, sigma: float, keep_dims: None | list[int] = None):
        """
        Initialize a FourierFeatureEmbedding module.

        Args:
            in_features (int):
                Total number of input features expected by the module.
            m (int):
                Number of random Fourier features per embedded dimension.
                The embedding produces `2 * m` output features (cosine and sine).
            sigma (float):
                Standard deviation for the Gaussian distribution used to sample
                the projection matrix `B` of shape `(len(embed_dims), m)`.
            keep_dims (list of int or None, optional):
                Indices of input dimensions to pass through unchanged.
                If `None`, all input features are Fourier-embedded.
                If provided, all dimensions *not* in `keep_dims` are embedded.
                Must not include all input dimensions, otherwise an error is raised.
        """
        super().__init__()

        self.in_features = in_features
        self.m = m
        self.sigma = sigma
        self.keep_dims = None if keep_dims is None else keep_dims[:]
        
        if keep_dims is None:
            embed_dims = list(range(in_features))
        else:
            embed_dims = [i for i in range(in_features) if i not in keep_dims]
            if len(embed_dims) == 0:
                raise ValueError("'keep_dims' list must not cover all the input features!")
        self.embed_dims = embed_dims

        self.register_buffer("B", torch.randn(len(embed_dims), m) * sigma)

        self.out_features = 2*m
        if  isinstance(self.keep_dims, list):
            self.out_features += len(self.keep_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fourier feature embedding for input `x`.

        For the dimensions specified by `embed_dims`, computes::

            xB = x[:, embed_dims] @ B
            output = concat( cos(xB), sin(xB), x[:, keep_dims]? )

        Args:
            x (Tensor):
                Input tensor of shape `(N, in_features)`.

        Returns:
            Tensor:
                Embedded tensor of shape `(N, out_features)`.
        """
        x_embed = x[:, self.embed_dims]
        xB = torch.matmul(x_embed, self.B)
        result = torch.cat( (torch.cos(xB), torch.sin(xB)), dim=1 )
        
        if self.keep_dims is None:
            return result
        else:
            return torch.cat((result, x[:, self.keep_dims]), dim=1)

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

        self.register_buffer('_coeffs', torch.tensor([
            k * w for w in [self.omega[i] for i in self.embed_dims] for k in range(1, self.m+1)]))

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