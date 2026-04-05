import torch

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