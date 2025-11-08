from abc import ABC, abstractmethod

import torch
import numpy as np
from scipy.special import jn_zeros, j0, j1

from NN_pytorch_BVP.pinn import *

# --- ИЕРАРХИЯ КЛАССОВ ДЛЯ КРАЕВОЙ ЗАДАЧИ ---
class BVP(ABC):
    # Свойство, возвращающее текстовое описание краевой задачи. Нужно для логов и сохранения результатов
    @property
    @abstractmethod
    def description(self) -> str: pass

    # Свойство, возвращающее полную размерность задачи. Например, задача u_t = u_xx + u_yy имеет полную размерность 3
    @property
    def problem_dim(self) -> int:
        spatial_dim = self.spatial_dim if isinstance(self, ISpatial) else 0
        temporal_dim = 1 if isinstance(self, ITemporal) else 0
        return spatial_dim + temporal_dim

    @abstractmethod
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor: pass
    
    @abstractmethod
    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor: pass    # model в параметрах метода это, конечно, костыль, но другого адекватного решения, которое не потребует переделывать половину кода и вводить новые абстракции, я пока не нашёл
    
    @abstractmethod
    def u_exact(self, x:torch.Tensor) -> torch.Tensor: pass


# Имитация интерфейсов (как в Java) - полностью абстрактных классов. 
# В Java один класс может наследоваться только от одного другого класса (унарное наследование) и от произвольного количества интерфейсов. 
# Интерфейсы в Java решают проблемы множественного наследования
class ISpatial(ABC):
    @property
    @abstractmethod
    def spatial_domain(self) -> list[float]: pass

    # Свойство, возвращающее пространственную размерность задачи
    @property
    @abstractmethod
    def spatial_dim(self) -> int: pass
    
    @abstractmethod
    def sample_bc(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor: pass

    @abstractmethod
    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor: pass

class ITemporal(ABC):
    @property
    @abstractmethod
    def temporal_domain(self) -> list[float]: pass

    @abstractmethod
    def sample_ic(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor: pass

    # Возвращает кортеж с двумя тензорами. Первый - это вычет начального условия u = f(x), где x принадлежит границе.
    # Второй - это вычет начального условия du/dt(x) = g(x)
    @abstractmethod
    def get_res_ic(self, model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: pass

# --- КЛАСС, ЗАДАЮЩИЙ КОНКРЕТНУЮ КРАЕВУЮ ЗАДАЧУ ---
class ODE1D_Phillipov756_Altered(BVP, ISpatial):
    @property
    def description(self) -> str:
        return """ --- ОБЫКНОВЕННОЕ ОДНОМЕРНОЕ НЕОДНОРОДНОЕ ДИФФЕРЕНЦИАЛЬНОЕ УРАВНЕНИЕ ---
Основано на ( Филиппов ) Сборник задач по дифференциальным уравнениям. Стр. 72, №756
u'' + u = 2x - pi,    0 < x < pi,
u  = 0,               x = 0,
u' = 0,               x = pi.
u_exact = pi*cos(x) + 2*sin(x) + 2*x - pi"""  

    @property
    def spatial_dim(self) -> int:
        return 1

    def __init__(self, spatial_domain, scheme='uniform', sobol_engine=None):
        super().__init__()
        self._spatial_domain = spatial_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine           

    @property
    def spatial_domain(self): 
        return self._spatial_domain

    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return sample_points_1D(self.spatial_domain, n, scheme=self.scheme, sobol_engine=self.sobol_engine).to(device)
    
    def sample_bc(self, n=2, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.tensor(self.spatial_domain, device=device).unsqueeze(1)

    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]
        return d2u_dx2 + u - 2.0 * x + torch.pi

    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        return u * torch.tensor([[1.0], [0.0]], device=u.device) + du_dx * torch.tensor([[0.0], [1.0]], device=u.device)
    
    def u_exact(self, x):
        return torch.pi * torch.cos(x) + 2.0 * torch.sin(x) + 2.0 * x - torch.pi

class HarmonicOscillatorUnderdamped1D(BVP, ITemporal):
    delta = 2
    omega0 = 20

    @property
    def description(self) -> str:
        return """ --- ОДНОМЕРНЫЙ ЗАТУХАЮЩИЙ ГАРМОНИЧЕСКИЙ ОСЦИЛЛЯТОР ---
u'' + 2 * delta * u' + omega0^2 * t = 0, где delta = mu/(2m), omega0 = sqrt(k/m)   (случай с затуханиями: delta < omega0)
u(0)  = 1,
u'(0) = 0.
u_exact(t) = 2 * A * exp(-delta * t) * cos(phi + omega * t), 
где omega = sqrt(omega0^2 - delta^2), 
    phi   = arctg(-delta/omega), 
    A     = 1 / (2 * cos(phi))"""

    def __init__(self, temporal_domain, scheme='uniform', sobol_engine=None):
        super().__init__()
        self._temporal_domain = temporal_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine 

    @property
    def temporal_domain(self): 
        return self._temporal_domain

    def oscillator(self, delta, omega0, t):
        """Defines the analytical solution to the 1D underdamped harmonic oscillator problem.
        Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
        assert delta < omega0
        omega = np.sqrt(omega0**2 - delta**2)
        phi = np.arctan(-delta / omega)
        A = 1.0 / ( 2.0 * np.cos(phi) )
        return 2.0 * A * torch.exp(-delta * t) * torch.cos(phi + omega * t)
    
    def u_exact(self, x):
        return self.oscillator(self.delta, self.omega0, x)
    
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return sample_points_1D(self.temporal_domain, n, scheme=self.scheme, sobol_engine=self.sobol_engine).to(device)

    def sample_ic(self, n=2, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.ones((1, 1), device=device) * self._temporal_domain[0]
    
    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]
        return d2u_dx2 + 2 * self.delta * du_dx + (self.omega0)**2 * u
    
    def get_res_ic(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        return u - torch.tensor([[1.0]], device=u.device), du_dx

class HeatEquation1D_Pozharsky47_5(BVP, ISpatial, ITemporal):
    @property
    def description(self) -> str:
        return """ --- УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ. ОДНОМЕРНОЕ НЕОДНОРОДНОЕ СО СМЕШАННЫМИ ОДНОРОДНЫМИ К.У. ---
Пожарский. Методическое пособие (6 семестр). Страница 158. Пример 47.5
u_t = u_xx + 2 * sin(t) * cos(x),    0 < x < pi/2, 0 < t < 2 * pi,
u_x = 0,                             x = 0,
u = 0,                               x = pi/2,
u = cos(x) - cos(5*x),               t = 0.
u_exact = (2 * exp(-t) + sin(t) - cos(t)) * cos(x) - exp(-25 * t) * cos(5 * x)"""  

    @property
    def spatial_dim(self) -> int:
        return 1

    def __init__(self, spatial_domain, temporal_domain, scheme='uniform', sobol_engine=None):
        super().__init__()
        self._spatial_domain = spatial_domain
        self._temporal_domain = temporal_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine 

    @property
    def spatial_domain(self): 
        return self._spatial_domain
    
    @property
    def temporal_domain(self): 
        return self._temporal_domain
    
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_2D( [t_min, x_min, t_max, x_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def sample_bc(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        p1 = sample_points_2D( [t_min, x_min, t_max, x_min], n, self.scheme, sobol_engine=self.sobol_engine, device=device )
        p2 = sample_points_2D( [t_min, x_max, t_max, x_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )
        return torch.cat( (p1, p2), dim=0 )
    
    def sample_ic(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_2D( [t_min, x_min, t_min, x_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)

        tmp = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = tmp[:, 0:1]
        u_x = tmp[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]

        return u_xx + 2 * torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2]) - u_t
    
    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]

        # Входной тензор x содержит координаты точек коллокации, принадлежащих разным границам.
        # Сначала надо вычислить индексы точек, принадлежащих границе x = x_min и x = x_max
        x_min, x_max = self.spatial_domain
        id1 = torch.nonzero( torch.abs(x[:, 1] - x_min) < 1e-4 ).squeeze()
        id2 = torch.nonzero( torch.abs(x[:, 1] - x_max) < 1e-4 ).squeeze()

        # На всякий случай проверяем, что 1) никакой элемент из id1 не содержится в id2 и
        # 2) объединение id1 и id2 содержит все индексы от 0 до n-1, где n - кол-во точек
        _, cnt = torch.cat( (id1, id2) ).unique(return_counts=True)
        assert (cnt > 1).sum() == 0, "Subsets overlap!"
        assert id1.numel() + id2.numel() == x.shape[0], "Missing/duplicate rows!"

        res = torch.empty(u_x.shape, dtype=u_x.dtype, device=u_x.device)
        res[id1] = u_x[id1]
        res[id2] = u[id2]
        return res
    
    def get_res_ic(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        return torch.cos(x[:, 1:2]) - torch.cos(5.0 * x[:, 1:2]) - u, torch.zeros_like(u)
    
    def u_exact(self, tx):
        t = tx[:, 0:1]
        x = tx[:, 1:2]
        return ( 2.0 * torch.exp(-t) + torch.sin(t) - torch.cos(t) ) * torch.cos(x) - torch.exp(-25.0 * t) * torch.cos(5.0 * x)

class HeatEquation2D_Pozharsky17(BVP, ISpatial, ITemporal):
    @property
    def description(self) -> str:
        return """ --- УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ. ДВУМЕРНОЕ НЕОДНОРОДНОЕ СО СМЕШАННЫМИ ОДНОРОДНЫМИ К.У. ---
Пожарский. Методическое пособие (6 семестр). стр. 183
u_t = u_xx + u_yy + t * sin(x),    0 < x < 2*pi, 0 < y < 1, 0 < t < 1,
u = 0,                             x = 0, 0 <= y <= 1,
u = 0,                             x = 2*pi, 0 <= y <= 1,
u_y = 0,                           0 <= x <= 2*pi, y = 0,
u_y = 0,                           0 <= x <= 2*pi, y = 1,
u = 2 * sin(2*x)                   t = 0.
u_exact = (t - 1 + exp(-t)) * sin(x) + 2 * exp(-4*t) * sin(2*x)"""  

    @property
    def spatial_dim(self) -> int:
        return 2

    def __init__(self, spatial_domain, temporal_domain, scheme='uniform', sobol_engine=None):
        super().__init__()
        self._spatial_domain = spatial_domain
        self._temporal_domain = temporal_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine 

    @property
    def spatial_domain(self): 
        return self._spatial_domain
    
    @property
    def temporal_domain(self): 
        return self._temporal_domain
    
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_3D( [t_min, x_min, y_min, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def sample_bc(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        p1 = sample_points_3D( [t_min, x_min, y_min, t_max, x_max, y_min], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # y = y_min
        p2 = sample_points_3D( [t_min, x_max, y_min, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # x = x_max
        p3 = sample_points_3D( [t_min, x_min, y_max, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # y = y_max
        p4 = sample_points_3D( [t_min, x_min, y_min, t_max, x_min, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # x = x_min
        return torch.cat( (p1, p2, p3, p4), dim=0 )
    
    def sample_ic(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_3D( [t_min, x_min, y_min, t_min, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)

        tmp = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = tmp[:, 0:1]
        u_x = tmp[:, 1:2]
        u_y = tmp[:, 2:3]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 2:3]

        return u_xx + u_yy + x[:, 0:1] * torch.sin( x[:, 1:2] ) - u_t
    
    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        u_y = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 2:3]

        # Входной тензор x содержит координаты точек коллокации, принадлежащих разным границам.
        # Сначала надо вычислить индексы точек, принадлежащих границе x = x_min и x = x_max
        x_min, y_min, x_max, y_max = self.spatial_domain
        eps = 1e-6
        id1 = torch.nonzero( torch.abs(x[:, 1] - x_min) < eps ).squeeze()
        id2 = torch.nonzero( torch.abs(x[:, 1] - x_max) < eps ).squeeze()
        id3 = torch.nonzero( torch.abs(x[:, 2] - y_min) < eps ).squeeze()
        id4 = torch.nonzero( torch.abs(x[:, 2] - y_max) < eps ).squeeze()

        # На всякий случай проверяем, что 1) никакой элемент из одного id не содержится в другом и
        # 2) объединение всех id содержит все индексы от 0 до n-1, где n - кол-во точек
        #_, cnt = torch.cat( (id1, id2, id3, id4) ).unique(return_counts=True)
        #assert (cnt > 1).sum() == 0, "Subsets overlap!"
        #assert id1.numel() + id2.numel() + id3.numel() + id4.numel() == x.shape[0], "Missing/duplicate rows!"

        res = torch.empty(u_y.shape, dtype=u_y.dtype, device=u_y.device)
        res[id1] = u[id1]
        res[id2] = u[id2]
        res[id3] = u_y[id3]
        res[id4] = u_y[id4]
        return res
    
    def get_res_ic(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        return u - 2.0 * torch.sin(2.0 * x[:, 1:2]), torch.zeros_like(u)
    
    def u_exact(self, txy):
        t = txy[:, 0:1]
        x = txy[:, 1:2]
        y = txy[:, 2:3]
        return (t - 1.0 + torch.exp(-t)) * torch.sin(x) + 2.0 * torch.exp(-4.0 * t) * torch.sin(2.0 * x)

class HeatEquationCircle_Bogolubov212_5(BVP, ISpatial, ITemporal):
    @property
    def description(self) -> str:
        return """ --- УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ. ДВУМЕРНОЕ ОДНОРОДНОЕ К.У. ДИРИХЛЕ. В КРУГЕ. ---
Боголюбов, Кравцов. Задачи по математической физике. Страница 212, задача №5
u_t = u_xx + u_yy,    0 < r < 5,    0 <= phi <= 2*pi
u = 0,    t = 0
u = 8,    r = 5.
u_exact = 8 + 16 * sum_n=1^\inf j0(mu_n * r / 5) * exp( -(mu_n/5)^2 * t ) / ( -mu_n * j1(mu_n) ),
    где j0(mu_n) = 0,    n = 1, 2, 3, ...)"""  

    @property
    def spatial_dim(self) -> int:
        return 2

    def __init__(self, spatial_domain, temporal_domain, scheme='uniform', sobol_engine=None):
        super().__init__()
        self._spatial_domain = spatial_domain    # [r_min, phi_min, r_max, phi_max]
        self._temporal_domain = temporal_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine 

        self.mu_n = torch.tensor(jn_zeros(0, 50))

    @property
    def spatial_domain(self): 
        return self._spatial_domain
    
    @property
    def temporal_domain(self): 
        return self._temporal_domain
    
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        r_min, phi_min, r_max, phi_max = self.spatial_domain
        t_min, t_max = self.temporal_domain

        t_r_phi = sample_points_3D( [t_min, r_min, phi_min, t_max, r_max, phi_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )
        txy = torch.empty_like(t_r_phi)
        txy[:, 0] = t_r_phi[:, 0]
        txy[:, 1] = t_r_phi[:, 1] * torch.cos(t_r_phi[:, 2])
        txy[:, 2] = t_r_phi[:, 1] * torch.sin(t_r_phi[:, 2])

        return txy

    def sample_bc(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        r_min, phi_min, r_max, phi_max = self.spatial_domain
        t_min, t_max = self.temporal_domain

        t_r_phi = sample_points_3D( [t_min, r_max, phi_min, t_max, r_max, phi_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )
        txy = torch.empty_like(t_r_phi)
        txy[:, 0] = t_r_phi[:, 0]
        txy[:, 1] = t_r_phi[:, 1] * torch.cos(t_r_phi[:, 2])
        txy[:, 2] = t_r_phi[:, 1] * torch.sin(t_r_phi[:, 2])

        return txy
    
    def sample_ic(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        r_min, phi_min, r_max, phi_max = self.spatial_domain
        t_min, t_max = self.temporal_domain

        t_r_phi = sample_points_3D( [t_min, r_min, phi_min, t_min, r_max, phi_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )
        txy = torch.empty_like(t_r_phi)
        txy[:, 0] = t_r_phi[:, 0]
        txy[:, 1] = t_r_phi[:, 1] * torch.cos(t_r_phi[:, 2])
        txy[:, 2] = t_r_phi[:, 1] * torch.sin(t_r_phi[:, 2])

        return txy

    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)

        tmp = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = tmp[:, 0:1]
        u_x = tmp[:, 1:2]
        u_y = tmp[:, 2:3]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 2:3]

        return u_xx + u_yy - u_t
    
    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor:
        return model(x) - 8.0
    
    def get_res_ic(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        return u, torch.zeros_like(u)
    
    def u_exact(self, txy):
        t = txy[:, 0:1].detach().cpu()
        r = torch.sqrt( txy[:, 1:2]**2 + txy[:, 2:3]**2 ).detach().cpu()

        n = 20
        mu_n = self.mu_n
 
        tmp = torch.zeros_like(r)
        for i in range(n):
            tmp += j0(mu_n[i] * r * 0.2) * torch.exp( -(mu_n[i] * 0.2)**2 * t ) / ( -mu_n[i] * j1(mu_n[i]) )
        return tmp.to(txy.device) * 16.0 + 8.0

class HeatEquation2D_Pozharsky50_3(BVP, ISpatial, ITemporal):
    @property
    def description(self) -> str:
        return """--- УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ. ДВУМЕРНОЕ ОДНОРОДНОЕ С ОДНОРОДНЫМИ К.У ДИРИХЛЕ. ---
Пожарский. Методическое пособие (6 семестр). стр. 177, задача 50.3
u_t = u_xx + u_yy,    0 < x < pi, 0 < y < pi,
u = 0,                x = 0, 
u = 0,                x = pi, 
u = 0,                y = 0,
u = 0,                y = pi,
u = 1                 t = 0.
u_exact = \sum_{n=1}^\inf \sum_{p=1}^\inf 4 / (pi^2 * n * p) (1 - (-1)^n) * (1 - (-1)^p) * exp(-(n^2 + p^2)*t) * sin(n*x) * sin(p*y)""" 

    @property
    def spatial_dim(self) -> int:
        return 2

    def __init__(self, spatial_domain, temporal_domain, scheme='uniform', sobol_engine=None):
        super().__init__()
        self._spatial_domain = spatial_domain
        self._temporal_domain = temporal_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine 

    @property
    def spatial_domain(self): 
        return self._spatial_domain
    
    @property
    def temporal_domain(self): 
        return self._temporal_domain
    
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_3D( [t_min, x_min, y_min, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def sample_bc(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        p1 = sample_points_3D( [t_min, x_min, y_min, t_max, x_max, y_min], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # y = y_min
        p2 = sample_points_3D( [t_min, x_max, y_min, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # x = x_max
        p3 = sample_points_3D( [t_min, x_min, y_max, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # y = y_max
        p4 = sample_points_3D( [t_min, x_min, y_min, t_max, x_min, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # x = x_min
        return torch.cat( (p1, p2, p3, p4), dim=0 )
    
    def sample_ic(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_3D( [t_min, x_min, y_min, t_min, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)

        tmp = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = tmp[:, 0:1]
        u_x = tmp[:, 1:2]
        u_y = tmp[:, 2:3]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 2:3]

        return u_xx + u_yy - u_t
    
    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        #u_y = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 2:3]

        # Входной тензор x содержит координаты точек коллокации, принадлежащих разным границам.
        # Сначала надо вычислить индексы точек, принадлежащих границе x = x_min и x = x_max
        x_min, y_min, x_max, y_max = self.spatial_domain
        eps = 1e-6
        id1 = torch.nonzero( torch.abs(x[:, 1] - x_min) < eps ).squeeze()
        id2 = torch.nonzero( torch.abs(x[:, 1] - x_max) < eps ).squeeze()
        id3 = torch.nonzero( torch.abs(x[:, 2] - y_min) < eps ).squeeze()
        id4 = torch.nonzero( torch.abs(x[:, 2] - y_max) < eps ).squeeze()

        # На всякий случай проверяем, что 1) никакой элемент из одного id не содержится в другом и
        # 2) объединение всех id содержит все индексы от 0 до n-1, где n - кол-во точек
        #_, cnt = torch.cat( (id1, id2, id3, id4) ).unique(return_counts=True)
        #assert (cnt > 1).sum() == 0, "Subsets overlap!"
        #assert id1.numel() + id2.numel() + id3.numel() + id4.numel() == x.shape[0], "Missing/duplicate rows!"

        res = torch.empty(u.shape, dtype=u.dtype, device=u.device)
        res[id1] = u[id1]
        res[id2] = u[id2]
        res[id3] = u[id3]
        res[id4] = u[id4]
        return res
    
    def get_res_ic(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        return u - 1.0, torch.zeros_like(u)
    
    def u_exact(self, txy):
        t = txy[:, 0:1]
        x = txy[:, 1:2]
        y = txy[:, 2:3]

        N = 100
        P = 100

        res = 0
        for n in range(1, N+1, 2):
            for p in range(1, P+1, 2):
                res += torch.exp(-(n*n+p*p)*t)*torch.sin(n*x)*torch.sin(p*y)/(n*p)
        res *= 16/np.pi/np.pi
        return res

class HeatEquation1D_Wang(BVP, ISpatial, ITemporal):
    @property
    def description(self) -> str:
        return """--- УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ. ОДНОМЕРНОЕ ОДНОРОДНОЕ С ОДНОРОДНЫМИ К.У. ДИРИХЛЕ. СПЕЦИАЛЬНОЕ ВЫСОКОЧАСТОТНОЕ ---
Wang. On the eigenvector bias of fourier feature networks - From regression to solving multi-scale PDEs with physics-informed neural networks. page 15
u_t = 1 / (gamma*pi)^2 * u_xx,    0 < x < 1,
u = 0,                            x = 0,
u = 0,                            x = 1,
u = sin(gamma*pi*x)               t = 0.
u_exact = exp(-t) * sin(gamma*pi*x),
где gamma - целочисленный параметр"""

    @property
    def spatial_dim(self) -> int:
        return 1

    def __init__(self, spatial_domain, temporal_domain, scheme='uniform', sobol_engine=None, gamma=2):
        super().__init__()
        self._spatial_domain = spatial_domain
        self._temporal_domain = temporal_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine 
        self.gamma = gamma

    @property
    def spatial_domain(self): 
        return self._spatial_domain
    
    @property
    def temporal_domain(self): 
        return self._temporal_domain
    
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_2D( [t_min, x_min, t_max, x_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def sample_bc(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        p1 = sample_points_2D( [t_min, x_min, t_max, x_min], n, self.scheme, sobol_engine=self.sobol_engine, device=device )
        p2 = sample_points_2D( [t_min, x_max, t_max, x_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )
        return torch.cat( (p1, p2), dim=0 )
    
    def sample_ic(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_2D( [t_min, x_min, t_min, x_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)

        tmp = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = tmp[:, 0:1]
        u_x = tmp[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]

        return u_xx - u_t * (self.gamma * torch.pi)**2
    
    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)

        # Входной тензор x содержит координаты точек коллокации, принадлежащих разным границам.
        # Сначала надо вычислить индексы точек, принадлежащих границе x = x_min и x = x_max
        x_min, x_max = self.spatial_domain
        id1 = torch.nonzero( torch.abs(x[:, 1] - x_min) < 1e-4 ).squeeze()
        id2 = torch.nonzero( torch.abs(x[:, 1] - x_max) < 1e-4 ).squeeze()

        # На всякий случай проверяем, что 1) никакой элемент из id1 не содержится в id2 и
        # 2) объединение id1 и id2 содержит все индексы от 0 до n-1, где n - кол-во точек
        _, cnt = torch.cat( (id1, id2) ).unique(return_counts=True)
        assert (cnt > 1).sum() == 0, "Subsets overlap!"
        assert id1.numel() + id2.numel() == x.shape[0], "Missing/duplicate rows!"

        res = torch.empty(u.shape, dtype=u.dtype, device=u.device)
        res[id1] = u[id1]
        res[id2] = u[id2]
        return res
    
    def get_res_ic(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        return torch.sin(self.gamma * torch.pi * x[:, 1:2]) - u, torch.zeros_like(u)
    
    def u_exact(self, tx):
        t = tx[:, 0:1]
        x = tx[:, 1:2]
        return torch.exp(-t) * torch.sin(self.gamma * torch.pi * x)

class HeatEquation2D_Custom1(BVP, ISpatial, ITemporal):
    @property
    def description(self) -> str:
        return """--- УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ. ДВУМЕРНОЕ НЕОДНОРОДНОЕ С ОДНОРОДНЫМИ К.У. ДИРИХЛЕ. СПЕЦИАЛЬНОЕ ВЫСОКОЧАСТОТНОЕ ---
Подобрано самостоятельно
u_t = alpha * (u_xx + u_yy) + f(x, y, t),    0 < x < 1, 0 < y < 1,
u = 0,                                       x = 0,
u = 0,                                       x = 1,
u = 0,                                       y = 0,
u = 0,                                       y = 1,
u = u_3_4 + 0.5 * u_8_6                      t = 0.
u_exact = exp(-25*alpha*pi^2*t) * u_3_4 + 0.5 * exp(-100*alpha*pi^2*t) * u_8_6 + c*(1 - exp(-5*t)) * u_2_3,
где alpha = 0.005 - положительное число,
    c = 1.0 - параметр, вещественное число, 
    u_n_p = sin(n * pi * x) * sin(p * pi * y),
    f(x, y, t) = с * [ 13*alpha*pi^2 + (5 - 13*alpha*pi^2) * exp(-5*t) ] * u_2_3"""  

    @property
    def spatial_dim(self) -> int:
        return 2

    def __init__(self, spatial_domain, temporal_domain, scheme='uniform', sobol_engine=None, alpha=0.005, c=1.0):
        super().__init__()
        self._spatial_domain = spatial_domain
        self._temporal_domain = temporal_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine 
        self.alpha = alpha
        self.c = c

    @property
    def spatial_domain(self): 
        return self._spatial_domain
    
    @property
    def temporal_domain(self): 
        return self._temporal_domain
    
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_3D( [t_min, x_min, y_min, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def sample_bc(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        p1 = sample_points_3D( [t_min, x_min, y_min, t_max, x_max, y_min], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # y = y_min
        p2 = sample_points_3D( [t_min, x_max, y_min, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # x = x_max
        p3 = sample_points_3D( [t_min, x_min, y_max, t_max, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # y = y_max
        p4 = sample_points_3D( [t_min, x_min, y_min, t_max, x_min, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device ) # x = x_min
        return torch.cat( (p1, p2, p3, p4), dim=0 )
    
    def sample_ic(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, y_min, x_max, y_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_3D( [t_min, x_min, y_min, t_min, x_max, y_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)

        tmp = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = tmp[:, 0:1]
        u_x = tmp[:, 1:2]
        u_y = tmp[:, 2:3]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 2:3]
        
        tmp = 13 * self.alpha * torch.pi**2
        return u_t - self.alpha * (u_xx + u_yy) - self.c * (tmp + (5 - tmp) * torch.exp(-5*x[:, 0:1])) * torch.sin(2 * torch.pi * x[:, 1:2]) * torch.sin(3 * torch.pi * x[:, 2:3])
    
    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        u_y = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 2:3]

        # Входной тензор x содержит координаты точек коллокации, принадлежащих разным границам.
        # Сначала надо вычислить индексы точек, принадлежащих границе x = x_min и x = x_max
        x_min, y_min, x_max, y_max = self.spatial_domain
        eps = 1e-6
        id1 = torch.nonzero( torch.abs(x[:, 1] - x_min) < eps ).squeeze()
        id2 = torch.nonzero( torch.abs(x[:, 1] - x_max) < eps ).squeeze()
        id3 = torch.nonzero( torch.abs(x[:, 2] - y_min) < eps ).squeeze()
        id4 = torch.nonzero( torch.abs(x[:, 2] - y_max) < eps ).squeeze()

        # На всякий случай проверяем, что 1) никакой элемент из одного id не содержится в другом и
        # 2) объединение всех id содержит все индексы от 0 до n-1, где n - кол-во точек
        #_, cnt = torch.cat( (id1, id2, id3, id4) ).unique(return_counts=True)
        #assert (cnt > 1).sum() == 0, "Subsets overlap!"
        #assert id1.numel() + id2.numel() + id3.numel() + id4.numel() == x.shape[0], "Missing/duplicate rows!"

        res = torch.empty(u_y.shape, dtype=u_y.dtype, device=u_y.device)
        res[id1] = u[id1]
        res[id2] = u[id2]
        res[id3] = u[id3]
        res[id4] = u[id4]
        return res
    
    def get_res_ic(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        return torch.sin(3.0 * torch.pi * x[:, 1:2]) * torch.sin(4.0 * torch.pi * x[:, 2:3]) + \
            0.5*torch.sin(8.0 * torch.pi * x[:, 1:2])*torch.sin(6.0 * torch.pi * x[:, 2:3]) - u, torch.zeros_like(u)
    
    def u_exact(self, txy):
        t = txy[:, 0:1]
        x = txy[:, 1:2]
        y = txy[:, 2:3]

        u_3_4 = torch.sin(3 * torch.pi * x) * torch.sin(4 * torch.pi * y)
        u_8_6 = torch.sin(8 * torch.pi * x) * torch.sin(6 * torch.pi * y)
        u_2_3 = torch.sin(2 * torch.pi * x) * torch.sin(3 * torch.pi * y)
        a = self.alpha * torch.pi**2 * t
        
        return torch.exp(-25 * a) * u_3_4 + 0.5 * torch.exp(-100 * a) * u_8_6 + self.c * (1.0 - torch.exp(-5*t)) * u_2_3
    
class HeatEquation1D_PeriodicBC(BVP, ISpatial, ITemporal):
    @property
    def description(self) -> str:
        return """--- УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ. ОДНОМЕРНОЕ ОДНОРОДНОЕ С ПЕРИОДИЧЕСКИМИ К.У. ---
Подобрано самостоятельно
u_t = u_xx,    0 < x < 2*Pi,
u(0, t) = u(2*Pi, t),
u_x(0, t) = u_x(2*Pi, t),
u(x, 0) = sin(x) + 1/2 * cos(2*x).
u_exact = exp(-t) * sin(x) + exp(-4*t)/2 * cos(2*x)"""  

    @property
    def spatial_dim(self) -> int:
        return 1
    
    def __init__(self, spatial_domain, temporal_domain, scheme='uniform', sobol_engine=None):
        super().__init__()
        self._spatial_domain = spatial_domain
        self._temporal_domain = temporal_domain
        self.scheme = scheme
        if scheme == 'sobol' and (sobol_engine is None or not isinstance(sobol_engine, torch.quasirandom.SobolEngine)):
            raise ValueError("For 'sobol' scheme, a valid SobolEngine instance must be provided.")
        else:
            self.sobol_engine = sobol_engine 

    @property
    def spatial_domain(self): 
        return self._spatial_domain
    
    @property
    def temporal_domain(self): 
        return self._temporal_domain
    
    def sample_domain(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_2D( [t_min, x_min, t_max, x_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def sample_bc(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        n_half = n // 2 if n % 2 == 0 else n // 2 + 1    # нужно чтобы у каждой граничной точки была пара. Нужно чтобы вычислить вычеты на границах
        p1 = sample_points_2D( [t_min, x_min, t_max, x_min], n_half, self.scheme, sobol_engine=self.sobol_engine, device=device )
        p2 = p1.clone()
        p2[:, 1] += x_max - x_min
        return torch.cat( (p1, p2), dim=0 )

    def sample_ic(self, n, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x_min, x_max = self.spatial_domain
        t_min, t_max = self.temporal_domain
        return sample_points_2D( [t_min, x_min, t_min, x_max], n, self.scheme, sobol_engine=self.sobol_engine, device=device )

    def get_res_domain(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)

        tmp = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = tmp[:, 0:1]
        u_x = tmp[:, 1:2]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]

        return u_xx - u_t

    def get_res_bc(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]

        # Заранее известно, что у каждой граничной точки есть своя пара на противоположной границе, то есть:
        # 1) n = len(x) - чётное
        # 2) x[i, :] == x[i + n // 2, :], 0 <= i <= n // 2 - 1
        n = len(x)
        id1 = torch.zeros(n, dtype=torch.bool)
        id1[:n//2] = True

        res = torch.empty(u_x.shape, dtype=u_x.dtype, device=u_x.device)
        res[id1] = u[id1] - u[~id1] + u_x[id1] - u_x[~id1]
        res[~id1] = -u[id1] + u[~id1] - u_x[id1] + u_x[~id1]

        #res[id1] = torch.exp(-4*x[id1, 0:1]) / 2.0 - u[id1]
        #res[~id1] = torch.exp(-4*x[~id1, 0:1]) / 2.0 - u[~id1]
        
        return res

    def get_res_ic(self, model, x: torch.Tensor) -> torch.Tensor:
        u = model(x)
        return torch.sin(x[:, 1:2]) + torch.cos(2.0 * x[:, 1:2]) / 2.0 - u, torch.zeros_like(u)

    def u_exact(self, tx):
        t = tx[:, 0:1]
        x = tx[:, 1:2]
        return torch.exp(-t) * torch.sin(x) + torch.exp(-4*t) / 2.0 * torch.cos(2*x)