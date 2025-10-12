from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # корень проекта
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from NN_pytorch_BVP.pinn import *


device = 'cpu'
torch.manual_seed(2007)
model = MultilayerPerceptronWithFFE(
    layer_sizes=[3, 256, 256, 1], 
    init_scheme='glorot_normal', 
    activation_fn=nn.Tanh(),
    use_FFE=True,
    FFE_embed_dims=[],
    FFE_m=100,
    FFE_sigma=8.0
).to(device)

print("BEFORE SAVE:")
print(model.state_dict().keys())  # names
print(model(torch.tensor([0.0, 1.0, 2.0], device=device).reshape(1, -1)))
print()
model_path = Path.cwd() / 'scripts' / 'test.pth'
MultilayerPerceptronWithFFE.save(model, model_path)

model2 = MultilayerPerceptronWithFFE.load(model_path)
print("AFTER SAVE:")
print(model2.state_dict().keys())  # names
print(model2(torch.tensor([0.0, 1.0, 2.0], device=device).reshape(1, -1)))


# То есть сохраняем state_dict модели. 
# При этом надо помнить, где была модель на момент сохранения. 
# Плюс к этому какая-то канитель с доверенностью весов, поэтому надо указывать weights_only=False
# Плюс к этому перед загрузкой модели её надо создать с теми же параметрами, какие были у оригинальной модели, чтобы
# формы тензоров совпали





