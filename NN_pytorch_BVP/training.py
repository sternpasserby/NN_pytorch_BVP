from dataclasses import dataclass
from pathlib import Path
import time

import torch

from NN_pytorch_BVP.bvp import *
from NN_pytorch_BVP.formatted_table import FormattedTable


@dataclass
class TrainingParams:
    n_epochs: int
    n_train_points: int
    batch_size: int
    optimizer: any
    use_static_training_set: bool = False
    lambda_r: float = 1.0
    lambda_bc: float = None  
    lambda_ic: float = None

    use_grad_norm_weighting_scheme: bool = False
    grad_norm_weighting_freq :int = 25    # Частота обновления (в шагах градиентного спуска) скоростей обучения lambda.
                                          # Если частота 0, то схема взвешивания скоростей обучения не используется 
                                          # и лямбды не обновляются
                                          # Используется при обновлении лямбд: 
                                          # lambda_new = alpha * lambda_old + (1 − alpha) * lambda_new
    grad_norm_weighting_alpha: float = 0.9
    
    use_causal_weighting_scheme: bool = False
    causal_weighting_M: int = 10
    causal_weighting_epsilon: float = 1.0

# --- КЛАСС-ТРЕНЕР НЕЙРОННОЙ СЕТИ ---
# TODO: сделать полиморфным, то есть чтобы мог работать с задачами, зависящими от времени
class Coach():
    def __init__(self, model, problem_obj, results_dir: Path | None = None):
        self.model = model
        self.problem_obj = problem_obj
        if results_dir is not None:
            if not isinstance(results_dir, Path):
                raise ValueError("results_dir must be a Path object!")
        self.results_dir = results_dir

    def update(self, x_domain, x_bc, x_ic, optimizer, lambda_r=1.0, lambda_bc=1.0, lambda_ic=1.0, grad_norm_weighting_alpha=0.9, update_lambda=False, 
               use_causal_weighting_scheme=False, causal_weighting_M=10, causal_weighting_epsilon=1.0):
        model = self.model
        problem = self.problem_obj
        is_spatial = isinstance(problem, ISpatial)
        is_time_dependent = isinstance(problem, ITemporal)

        if use_causal_weighting_scheme:
            w = np.zeros(causal_weighting_M)
            w[0] = 1.0
            loss_r_i = []
        else:
            w = None
            loss_r_i = None

        # TODO: сделать use_causal_weighting_scheme универсальной, для любых размерностей тензора x_domain
        optimizer.zero_grad()
        if use_causal_weighting_scheme:
            # Вычисление loss_r_i (значений функции loss на временном отрезке от i * M до (i + 1) * M)
            M = causal_weighting_M
            boundaries = torch.linspace(problem.temporal_domain[0], problem.temporal_domain[1], M + 1, device=x_domain.device)
            inds = torch.bucketize(x_domain, boundaries[1:-1])

            for i in range(M):
                #mask = (x_domain > i * tau) & (x_domain <= (i + 1) * tau)
                #x_subdomain = x_domain[mask].view(-1, 1).clone().detach().to(x_domain.device).requires_grad_(x_domain.requires_grad)
                #loss_r_i[i] = torch.mean((problem.get_res_domain(model, x_subdomain))**2)
                loss_r_i.append( (problem.get_res_domain(model, x_domain[inds==i].view(-1, 1))).pow(2).mean() )
            
            # Вычисление значений весов
            loss_r_i = torch.stack(loss_r_i)
            cum_loss = loss_r_i.cumsum(dim=0).roll(1,0); cum_loss[0] = 0
            #for i in range(1, M, 1):
            #    tmp = 0.0
            #    for k in range(i):
            #        tmp += loss_r_i[k]
            #    w[i] = np.exp(-causal_weighting_epsilon * tmp.detach().cpu().numpy())

            # Вычисление самого loss_r
            w = torch.exp(-causal_weighting_epsilon * cum_loss)
            loss_r = (w * loss_r_i).mean()
            #tmp = 0.0
            #for i in range(M):
            #    tmp += w[i] * loss_r_i[i]
            #loss_r = tmp / M
        else: 
            loss_r = torch.mean((problem.get_res_domain(model, x_domain))**2)
        loss_r.backward(retain_graph=True)
        grad_loss_r_theta_norm = compute_grad_theta_norm(model)

        if is_spatial:
            optimizer.zero_grad()
            loss_bc = torch.mean((problem.get_res_bc(model, x_bc))**2)
            loss_bc.backward(retain_graph=True)
            grad_loss_bc_theta_norm = compute_grad_theta_norm(model)

        if is_time_dependent:
            optimizer.zero_grad()
            res_ic1, res_ic2 = problem.get_res_ic(model, x_ic)
            loss_ic = torch.mean(res_ic1**2) + torch.mean(res_ic2**2)
            loss_ic.backward(retain_graph=True)
            grad_loss_ic_theta_norm = compute_grad_theta_norm(model)

        if update_lambda:
            tmp = grad_loss_r_theta_norm
            if is_spatial: tmp += grad_loss_bc_theta_norm
            if is_time_dependent: tmp += grad_loss_ic_theta_norm

            alpha = grad_norm_weighting_alpha
            lambda_r_new = tmp / grad_loss_r_theta_norm
            lambda_r = alpha*lambda_r + (1.0 - alpha)*lambda_r_new
            if is_spatial: 
                lambda_bc_new = tmp / grad_loss_bc_theta_norm
                lambda_bc = alpha*lambda_bc + (1.0 - alpha)*lambda_bc_new
            if is_time_dependent: 
                lambda_ic_new = tmp / grad_loss_ic_theta_norm
                lambda_ic = alpha*lambda_ic + (1.0 - alpha)*lambda_ic_new
            
        loss = lambda_r * loss_r
        if is_spatial: loss += lambda_bc * loss_bc
        if is_time_dependent: loss += lambda_ic * loss_ic
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return (loss, loss_r, loss_bc if is_spatial else None, loss_ic if is_time_dependent else None, 
                lambda_r, lambda_bc if is_spatial else None, lambda_ic if is_time_dependent else None, w, loss_r_i)

    def train(self, training_params, verbose=True):
        # Назначение некоторых ярлыков для облегчения кода и ускорения доступа к полям данных соотв. объектов
        model = self.model
        device = next(model.parameters()).device
        problem = self.problem_obj
        is_spatial = isinstance(problem, ISpatial)
        is_time_dependent = isinstance(problem, ITemporal)
        results_dir = self.results_dir
        tm_filename = "training_metrics.txt"

        # Распаковка параметров обучения
        n_epochs =                       training_params.n_epochs
        n_train_points =                 training_params.n_train_points
        batch_size =                     training_params.batch_size
        optimizer =                      training_params.optimizer
        use_static_training_set =        training_params.use_static_training_set
        lambda_r =                       training_params.lambda_r
        lambda_bc =                      training_params.lambda_bc
        lambda_ic =                      training_params.lambda_ic
        use_grad_norm_weighting_scheme = training_params.use_grad_norm_weighting_scheme
        grad_norm_weighting_freq =       training_params.grad_norm_weighting_freq
        grad_norm_weighting_alpha =      training_params.grad_norm_weighting_alpha
        use_causal_weighting_scheme =    training_params.use_causal_weighting_scheme
        causal_weighting_M =             training_params.causal_weighting_M
        causal_weighting_epsilon =       training_params.causal_weighting_epsilon
        if lambda_bc is None and is_spatial:
            raise ValueError("For solving a boundary value problem with spatial componets, a valid lambda_bc value must be provided.")
        if lambda_ic is None and is_time_dependent:
            raise ValueError("For solving a non-static boundary value problem, a valid lambda_ic value must be provided.")

        # Инициализация таблицы со всеми метриками
        n_grad_steps = n_epochs * (n_train_points // batch_size)
        columns1 = [
            ("Epoch",          '{:6d} | '),
            ("loss_test",      '{:10.4e}   '),
            #("loss_r_test",    '{:12.4e}   '),
            #("loss_bc_test",   '{:12.4e}   '),
            #("loss_ic_test",   '{:12.4e} | '),
            ("loss",           '{:10.4e}   '),
            ("loss_r",         '{:10.4e}   '),
            ("loss_bc",        '{:10.4e}   '),
            ("loss_ic",        '{:10.4e} | '),
            ("lambda_r",       '{:10.4e}   '),
            ("lambda_bc",      '{:10.4e}   '),
            ("lambda_ic",      '{:10.4e} | '),
        ]
        if use_causal_weighting_scheme:
            columns2 = []
            for i in range(causal_weighting_M):
                columns2.append(("w_" + str(i), '{:10.4e}   '))
                if i == causal_weighting_M - 1:
                    columns2.append(("loss_r_" + str(i), '{:10.4e}  | '))
                else:
                    columns2.append(("loss_r_" + str(i), '{:10.4e}   '))
        else:
            columns2 = []
        columns3 = [
            ("err_l2",         '{:10.4e}   '),
            ("err_r_l2",       '{:10.4e}   '),
            ("err_bc_l2",      '{:10.4e}   '),
            ("err_ic_l2",      '{:10.4e} | '),
            ("err_inf",        '{:10.4e}   '),
            ("err_r_inf",      '{:10.4e}   '),
            ("err_bc_inf",     '{:10.4e}   '),
            ("err_ic_inf",     '{:10.4e} | '),
            ("rel_err_l2",     '{:12.2f}   '),
            ("rel_err_r_l2",   '{:12.2f}   '),
            ("rel_err_bc_l2",  '{:13.2f}   '),
            ("rel_err_ic_l2",  '{:13.2f} | '),
            ("rel_err_inf",    '{:13.2f}   '),
            ("rel_err_r_inf",  '{:13.2f}   '),
            ("rel_err_bc_inf", '{:14.2f}   '),
            ("rel_err_ic_inf", '{:14.2f} | '),
            ("res_l2",         '{:10.4e}   '),
            ("res_r_l2",       '{:10.4e}   '),
            ("res_bc_l2",      '{:10.4e}   '),
            ("res_ic_l2",      '{:10.4e} | '),
            ("res_inf",        '{:10.4e}   '),
            ("res_r_inf",      '{:10.4e}   '),
            ("res_bc_inf",     '{:10.4e}   '),
            ("res_ic_inf",     '{:10.4e} | '),
            ("time, sec",      '{:10.2f}')
        ]
        tm = FormattedTable(    # trainging metrics
            columns_info=columns1 + columns2 + columns3,
            n_rows=n_grad_steps
        )

        # Здесь некоторые столбцы таблицы отмечаются как скрытые, чтобы они не печатались при выводе
        if not is_spatial:
            tm.set_visibility(["lambda_bc", "loss_bc", "err_bc_l2", "err_bc_inf", "rel_err_bc_l2", "rel_err_bc_inf", "res_bc_l2", "res_bc_inf"], False)
        if not is_time_dependent:
            tm.set_visibility(["lambda_ic", "loss_ic", "err_ic_l2", "err_ic_inf", "rel_err_ic_l2", "rel_err_ic_inf", "res_ic_l2", "res_ic_inf"], False)

        # Генерация тестовых данных
        x_test_domain = problem.sample_domain(n=n_train_points, device=device).requires_grad_()
        u_test_domain = problem.u_exact(x_test_domain)
        if is_spatial:
            x_test_bc = problem.sample_bc(n=n_train_points, device=device).requires_grad_()
            u_test_bc = problem.u_exact(x_test_bc)
        if is_time_dependent:
            x_test_ic = problem.sample_ic(n=n_train_points, device=device).requires_grad_()
            u_test_ic = problem.u_exact(x_test_ic)
        u_test = torch.cat( [u_test_domain] + [u_test_bc] if is_spatial else [] + [u_test_ic] if is_time_dependent else [], dim=0)

        # Вывод преамбулы и заголовка таблицы с метриками в консоль и в текстовый файл
        preambule_str = "\n".join( [f"# {line}" for line in problem.description.splitlines()] ) + "\n" + \
                        "# \n" + \
                        "# device:                            " + str(device) + "\n" + \
                        "# \n" + \
                        "# TRAINING PARAMETERS:               " + "\n" + \
                        "# n_epochs:                          " + str(training_params.n_epochs) + "\n" + \
                        "# n_train_points:                    " + str(training_params.n_train_points) + "\n" + \
                        "# batch_size:                        " + str(training_params.batch_size) + "\n" + \
                        "# sampling_scheme:                   " + str(problem.scheme) + "\n" + \
                        "# learning rate (initial):           " + str(optimizer.param_groups[0]['lr']) + "\n" + \
                        "# optimizer:                         " + str(type(optimizer).__name__) + "\n" + \
                        "# use_static_training_set:           " + str(training_params.use_static_training_set) + "\n" + \
                        "# lambda_r (initial):                " + str(training_params.lambda_r) + "\n" + \
                        "# lambda_bc (initial):               " + str(training_params.lambda_bc) + "\n" + \
                        "# lambda_ic (initial):               " + str(training_params.lambda_ic) + "\n" + \
                        "# use_grad_norm_weighting:           " + str(training_params.use_grad_norm_weighting_scheme) + "\n" + \
                        "# grad_norm_weighing_freq:           " + str(training_params.grad_norm_weighting_freq) + " (in gradient descend steps)\n" + \
                        "# grad_norm_weighing_alpha:          " + str(training_params.grad_norm_weighting_alpha) + "\n" + \
                        "# use_causal_weighting_scheme:       " + str(training_params.use_causal_weighting_scheme) + "\n" + \
                        "# causal_weighting_M:                " + str(training_params.causal_weighting_M) + "\n" + \
                        "# causal_weighting_epsilon:          " + str(training_params.causal_weighting_epsilon) + "\n" + \
                        "# \n" + \
                        "# MODEL PARAMETERS:                  " + "\n" + \
                        "# layer_sizes:                       " + str(model.layer_sizes) + "\n" + \
                        "# init_scheme:                       " + str(model.init_scheme) + "\n" + \
                        "# activation_fn:                     " + str(model.activation_fn) + "\n" + \
                        "# use_FFE:                           " + str(model.use_FFE) + "\n" + \
                        "# FFE_embed_dims:                    " + str(model.FFE_embed_dims) + "\n" + \
                        "# FFE_m:                             " + str(model.FFE_m) + "\n" + \
                        "# FFE_sigma:                         " + str(model.FFE_sigma) + "\n"
        header_str = tm._header_as_string()
        if results_dir is not None:
            with open(results_dir / tm_filename, "w", encoding="utf-8") as f:
                f.write(preambule_str + '\n')
                f.write(header_str + '\n')
        if verbose:
            print(preambule_str)
            print(header_str)

        training_start_time = time.time()
        gs = -1    # gradient step number

        # Тренировочный цикл
        for epoch in range(n_epochs):
            epoch_start_time = time.time()

            # Генерация точек коллокации в случае, если это первая эпоха или нет флага использовать статический
            # набор точек коллокации
            if epoch == 0 or not use_static_training_set:
                x_train_domain = problem.sample_domain(n=n_train_points, device=device)
                if is_spatial: x_train_bc = problem.sample_bc(n=n_train_points, device=device)
                if is_time_dependent: x_train_ic = problem.sample_ic(n=n_train_points, device=device)
                update_lambda = False

            # Цикл по всем батчам текущей эпохи
            #indices = list(range(0, n_train_points, batch_size))
            #random.shuffle(indices)
            #for i in indices:
            for i in range(0, n_train_points, batch_size):
                gs += 1
                if use_grad_norm_weighting_scheme:
                    update_lambda = True if (gs + 1) % grad_norm_weighting_freq == 0 else False
                
                loss, loss_r, loss_bc, loss_ic, lambda_r, lambda_bc, lambda_ic, w, loss_r_i = self.update(
                    x_train_domain[i:i + batch_size].requires_grad_(), 
                    x_train_bc.requires_grad_() if is_spatial else None, 
                    x_train_ic.requires_grad_() if is_time_dependent else None, 
                    optimizer, lambda_r=lambda_r, lambda_bc=lambda_bc, lambda_ic=lambda_ic,
                    grad_norm_weighting_alpha=grad_norm_weighting_alpha, update_lambda=update_lambda,
                    use_causal_weighting_scheme=use_causal_weighting_scheme, causal_weighting_M=causal_weighting_M,
                    causal_weighting_epsilon=causal_weighting_epsilon
                )
                if use_causal_weighting_scheme:
                    for j in range(causal_weighting_M):
                        tm.set_value(f"w_{j:d}", gs, w[j].detach().cpu())
                        tm.set_value(f"loss_r_{j:d}", gs, loss_r_i[j].detach().cpu())
                
                # Вычисление ошибки err в области, на границе, для нач. условий, а также общей err
                with torch.no_grad():
                    u_pred_r = model.forward(x_test_domain)
                    err_domain =  torch.abs( u_test_domain - u_pred_r )
                    if is_spatial: 
                        u_pred_bc = model.forward(x_test_bc)
                        err_bc = torch.abs( u_test_bc - u_pred_bc )
                    if is_time_dependent:
                        u_pred_ic = model.forward(x_test_ic)
                        err_ic = torch.abs( u_test_ic - u_pred_ic )
                    u_pred = torch.cat( [u_pred_r] + [u_pred_bc] if is_spatial else [] + [u_pred_ic] if is_time_dependent else [] , dim=0)
                    err = torch.cat( [err_domain] + [err_bc] if is_spatial else [] + [err_ic] if is_time_dependent else [] , dim=0)
                
                # Вычисление вычета res в области, на границе, для нач. условий, а также общего res
                res_domain = problem.get_res_domain(model, x_test_domain)
                if is_spatial: 
                    res_bc = problem.get_res_bc(model, x_test_bc)
                else:
                    res_bc = torch.tensor([[-1.0]])
                if is_time_dependent: 
                    res_ic, res_ic2 = problem.get_res_ic(model, x_test_ic)
                else:
                    res_ic = torch.tensor([[-1.0]])
                res = torch.cat( [res_domain] + [res_bc] if is_spatial else [] + [res_ic] if is_time_dependent else [], dim=0)

                # Вычисление loss-функции на тестовых данных
                loss_r_test = torch.mean( res_domain**2 )
                if use_causal_weighting_scheme:
                    raise ValueError("Computing loss_r_test when causal weighting scheme is used has not been implemented yet!.")
                else:
                    loss_test = lambda_r * loss_r_test
                if is_spatial:
                    loss_bc_test = torch.mean( res_bc**2 )
                    loss_test += lambda_bc * loss_bc_test
                if is_time_dependent:
                    loss_ic_test = torch.mean( res_ic**2 ) + torch.mean( res_ic2**2 )
                    loss_test += lambda_ic * loss_ic_test

                # Высчисление метрик, запись их в таблицу результатов, вывод в файл и в консоль
                with torch.no_grad():
                    tm.set_value('Epoch', gs, epoch + 1)

                    tm.set_value('loss_test', gs, loss_test.detach().cpu())
                    #tm.set_value('loss_r_test', gs, loss_r_test.detach().cpu())
                    #tm.set_value('loss_bc_test', gs, loss_bc_test.detach().cpu())
                    #tm.set_value('loss_ic_test', gs, loss_ic_test.detach().cpu())
                    tm.set_value('loss', gs, loss.detach().cpu())
                    tm.set_value('lambda_r', gs, lambda_r)
                    tm.set_value('loss_r', gs, loss_r.detach().cpu())
                    err_l2 = torch.linalg.vector_norm(err, ord=2)
                    tm.set_value('err_l2', gs, err_l2.cpu() )
                    err_domain_l2 = torch.linalg.vector_norm(err_domain, ord=2)
                    tm.set_value('err_r_l2', gs, err_domain_l2.cpu())
                    err_inf = torch.linalg.vector_norm(err, ord=float('inf'))
                    tm.set_value('err_inf', gs, err_inf.cpu())
                    err_domain_inf = torch.linalg.vector_norm(err_domain, ord=float('inf'))
                    tm.set_value('err_r_inf', gs, err_domain_inf.cpu())
                    tm.set_value('rel_err_l2', gs, ( err_l2 / torch.linalg.vector_norm(u_test, ord=2) * 100 ).cpu() )
                    tm.set_value('rel_err_r_l2', gs, ( err_domain_l2 / torch.linalg.vector_norm(u_test_domain, ord=2) * 100 ).cpu() )
                    tm.set_value('rel_err_inf', gs, ( err_inf / torch.linalg.vector_norm(u_test, ord=float('inf')) * 100 ).cpu() )
                    tm.set_value('rel_err_r_inf', gs, ( err_domain_inf / torch.linalg.vector_norm(u_test_domain, ord=float('inf')) * 100 ).cpu() )
                    tm.set_value('res_l2', gs, torch.linalg.vector_norm(res, ord=2).cpu() )
                    tm.set_value('res_r_l2', gs, torch.linalg.vector_norm(res_domain, ord=2).cpu() )
                    tm.set_value('res_inf', gs, torch.linalg.vector_norm(res, ord=float('inf')).cpu() )
                    tm.set_value('res_r_inf', gs, torch.linalg.vector_norm(res_domain, ord=float('inf')).cpu() )
                    if is_spatial:
                        tm.set_value('lambda_bc', gs, lambda_bc)
                        tm.set_value('loss_bc', gs, loss_bc.detach().cpu())
                        err_bc_l2 = torch.linalg.vector_norm(err_bc, ord=2)
                        tm.set_value('err_bc_l2', gs, err_bc_l2.cpu())
                        err_bc_inf = torch.linalg.vector_norm(err_bc, ord=float('inf'))
                        tm.set_value('err_bc_inf', gs, err_bc_inf.cpu())
                        tm.set_value('rel_err_bc_l2', gs, ( err_bc_l2 / torch.linalg.vector_norm(u_test_bc, ord=2) * 100 ).cpu() )
                        tm.set_value('rel_err_bc_inf', gs, ( err_bc_inf / torch.linalg.vector_norm(u_test_bc, ord=float('inf')) * 100 ).cpu() )
                        tm.set_value('res_bc_l2', gs, torch.linalg.vector_norm(res_bc, ord=2).cpu() )
                        tm.set_value('res_bc_inf', gs, torch.linalg.vector_norm(res_bc, ord=float('inf')).cpu() )
                    if is_time_dependent:
                        tm.set_value('lambda_ic', gs, lambda_ic)
                        tm.set_value('loss_ic', gs, loss_ic.detach().cpu())
                        err_ic_l2 = torch.linalg.vector_norm( err_ic, ord=2 )
                        err_ic_inf = torch.linalg.vector_norm(err_ic, ord=float('inf'))
                        tm.set_value('err_ic_l2', gs, err_ic_l2.cpu())
                        tm.set_value('err_ic_inf', gs, err_ic_inf.cpu())
                        tm.set_value('rel_err_ic_l2', gs, ( err_ic_l2 / torch.linalg.vector_norm(u_test_ic, ord=2) * 100 ).cpu() )
                        tm.set_value('rel_err_ic_inf', gs, ( err_ic_inf / torch.linalg.vector_norm(u_test_ic, ord=float('inf')) * 100 ).cpu() )
                        tm.set_value('res_ic_l2', gs, torch.linalg.vector_norm(res_ic, ord=2).cpu() )
                        tm.set_value('res_ic_inf', gs, torch.linalg.vector_norm(res_ic, ord=float('inf')).cpu() )
                tm.set_value('time, sec', gs, time.time() - epoch_start_time)

                row_str = tm.row_as_string(gs)
            if results_dir is not None:
                with open(results_dir / tm_filename, "a", encoding="utf-8") as f:
                        f.write(row_str)
                        f.write('\n')
            if (epoch + 1) % 10 == 0 and verbose:
                    print(row_str)

        # Вывод времени, затраченного на обучение
        training_time = time.time() - training_start_time
        if verbose:
            print(
                f"Training took {int(training_time) // 3600} hour(s), " 
                f"{ (int(training_time) % 3600) // 60 } minute(s) and "
                f"{ (training_time % 3600) % 60:.2f} second(s)")

        return tm