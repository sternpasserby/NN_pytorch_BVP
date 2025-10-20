from pathlib import Path
import sys
import time
from types import FunctionType
import traceback

import torch

# Добавление корневой директории проекта в sys.path чтобы появилась
# возможность импортировать модули из NN_pytorch_BVP
ROOT = Path.cwd()  # корень проекта
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from NN_pytorch_BVP.pinn import *

# Создание папки для временных файлов, если её нет
temp_path = ROOT / 'tests' / 'temp'
temp_path.mkdir(exist_ok=True)

# ------------------------------------ ТЕСТЫ ------------------------------------

def test_save_load_results_cpu():
    device = 'cpu'
    model = MultilayerPerceptronWithFFE(
        layer_sizes=[3, 256, 512, 1], 
        init_scheme='glorot_normal', 
        activation_fn=nn.Tanh(),
        use_FFE=True,
        FFE_embed_dims=[],
        FFE_m=100,
        FFE_sigma=8.0
    ).to(device)

    model_path = temp_path / 'test_model_cpu.pth'
    MultilayerPerceptronWithFFE.save(model, model_path)
    model2 = MultilayerPerceptronWithFFE.load(model_path)

    x = torch.randn((10, 3), device=device)

    assert torch.all(model(x) == model2(x)), "Изменился результат работы модели после её загрузки с диска"

def test_save_load_results_cuda():
    device = 'cuda:0'
    model = MultilayerPerceptronWithFFE(
        layer_sizes=[3, 256, 512, 1], 
        init_scheme='glorot_normal', 
        activation_fn=nn.Tanh(),
        use_FFE=True,
        FFE_embed_dims=[],
        FFE_m=100,
        FFE_sigma=8.0
    ).to(device)

    model_path = temp_path / 'test_model_cuda0.pth'
    MultilayerPerceptronWithFFE.save(model, model_path)
    model2 = MultilayerPerceptronWithFFE.load(model_path)

    x = torch.randn((10, 3), device=device)
    
    assert torch.all(model(x) == model2(x)), "Изменился результат работы модели после её загрузки с диска"


# -------------------------------------------------------------------------------

def discover_tests(namespace):
    """Найти все функции в данном namespace, начинающиеся с test_* (обычно globals())"""
    tests_list = []
    for name, obj in list(namespace.items()):
        if name.startswith("test_") and isinstance(obj, FunctionType):
            tests_list.append((name, obj))
    return tests_list

def run_test(name: str, fn: FunctionType):
    start_time = time.time()
    try:
        fn()
    except AssertionError as e:
        duration = time.time() - start_time
        tb = traceback.format_exc()
        return False, "assertion", tb, duration
    except Exception as e: 
        duration = time.time() - start_time
        tb = traceback.format_exc()
        return False, "error", tb, duration
    else:
        duration = time.time() - start_time
        return True, "ok", None, duration
    
def main():
    # Обнаружение функций-тестов
    tests_list = discover_tests(globals())
    if len(tests_list) == 0:
        print("Тесты не найдены! (функции, начинающиеся с test_)")
        return 1
    
    # Запуск функций-тестов
    results = []
    for name, fn in tests_list:
        is_ok, error_kind, tb, duration = run_test(name, fn)
        results.append((name, is_ok, error_kind, tb, duration))

    # Вывод summary результатов тестирования
    n_total = len(results)
    n_passed = sum(1 for r in results if r[1])
    n_failed = sum(1 for r in results if not r[1])
    print("="*35 + " SUMMARY " + "="*35)
    print(f"Total: {n_total}, passed: {n_passed}, failed: {n_failed}")
    if n_passed > 0:
        print("\nPassed tests:")
        for name, is_ok, error_kind, tb, duration in results:
            if is_ok:
                print(f"- {name} ({duration:.3f} s)")
    if n_failed > 0:
        print("\nFailed tests:")
        for name, is_ok, error_kind, tb, duration in results:
            if not is_ok:
                print(f"- {name} ({error_kind}, {duration:.3f} s)")

    # Вывод более подробных результатов тестирования со стеком ошибок 
    # для каждого неуспешного теста
    verbose = True
    if verbose and n_failed > 0:
        print()
        print("="*79)
        print()
        for name, is_ok, error_kind, tb, duration in results:
            if not is_ok:
                print(f"[ {error_kind.upper()} ] {name} ({duration:.3f} s)\n{tb}")
    return 0 if n_failed == 0 else 2
    
if __name__ == "__main__":
    main()