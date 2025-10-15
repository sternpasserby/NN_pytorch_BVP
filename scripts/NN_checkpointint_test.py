from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # корень проекта
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from NN_pytorch_BVP.pinn import *


def test_save_load_results():
    device = 'cpu'
    model = MultilayerPerceptronWithFFE(
        layer_sizes=[3, 256, 256, 1], 
        init_scheme='glorot_normal', 
        activation_fn=nn.Tanh(),
        use_FFE=True,
        FFE_embed_dims=[],
        FFE_m=100,
        FFE_sigma=8.0
    ).to(device)

    model_path = Path.cwd() / 'scripts' / 'test.pth'
    MultilayerPerceptronWithFFE.save(model, model_path)
    model2 = MultilayerPerceptronWithFFE.load(model_path)

    x = torch.randn((10, 3), device=device)
    
    assert torch.all(model(x) == model2(x)), "Изменился результат работы модели после её загрузки с диска"


# То есть сохраняем state_dict модели. 
# При этом надо помнить, где была модель на момент сохранения. 
# Плюс к этому какая-то канитель с доверенностью весов, поэтому надо указывать weights_only=False
# Плюс к этому перед загрузкой модели её надо создать с теми же параметрами, какие были у оригинальной модели, чтобы
# формы тензоров совпали





