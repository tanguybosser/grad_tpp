import torch as th 
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os
import json
import pickle as pkl

from tpps.models.base.process import Process
from torch.linalg import vector_norm
from copy import deepcopy

def detach(x: th.Tensor) -> th.Tensor:
    return x.cpu().detach().numpy()

def count_parameters(model: Process):
    enc, rest = 0,0
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'encoder' in n:
                enc += p.numel()
            else:
                rest += p.numel()
    enc_prop = enc/total
    rest_prop = rest/total 
    print(f'Total parameters: {total}')
    print(f'Encoder Total: {enc}')
    print(f'Rest Total: {rest}')
    print(f'Encoder Proportion: {enc_prop}')
    print(f'Rest Proportion: {rest_prop}')

def compute_grad_metrics(time_grads, mark_grads, dot_dic, sim_dic, tpi_dic):
    for k in time_grads.keys():
        g_t = time_grads[k]
        g_m = mark_grads[k]
        norm_g_t = vector_norm(g_t, ord=2)
        norm_g_m = vector_norm(g_m, ord=2)
        tpi = int(norm_g_m < norm_g_t)
        #Gradient magnitude similarity. 
        gms = (2 * vector_norm(g_t, ord=2) * vector_norm(g_m, ord=2))
        gms = gms/(th.square(vector_norm(g_t, ord=2)) + th.square(vector_norm(g_m, ord=2)))                
        #Avoids numerical instabilities 
        g_t[th.abs(g_t) < 1e-6] = 0
        g_m[th.abs(g_m) < 1e-6] = 0
        g_t = F.normalize(g_t, dim=0)
        g_m = F.normalize(g_m, dim=0)
        dot = float((g_t * g_m).sum(dim=0))
        dot_dic[k].append(dot)
        sim_dic[k].append(float(gms))
        tpi_dic[k].append(tpi)        
    return dot_dic, sim_dic, tpi_dic


def early_stopping(args, val_metrics, train_metrics_list,
                   early_stopping_var,
                   model, epoch, optimizer): 
    new_best, new_best_time, new_best_mark = is_model_best(val_metrics, 
                                                           early_stopping_var,
                                                           args)
    if not args.separate_training:
        if new_best:
            early_stopping_var['best_loss'] = val_metrics['loss_t'] + val_metrics['loss_m'] + val_metrics['loss_w']
            early_stopping_var['cnt_wait'] = 0
            early_stopping_var['best_state'] = deepcopy(model.state_dict())
        else:
            early_stopping_var['cnt_wait'] += 1
        if early_stopping_var['cnt_wait'] > args.patience:
            print("Early stopping! Stopping at epoch {}".format(str(epoch)), flush=True)
            early_stopping_var['early_stop'] = True
    else:
        if new_best_time and early_stopping_var['cnt_wait_time'] < args.patience:
            early_stopping_var['best_loss_time'] = val_metrics['loss_t'] + val_metrics['loss_w']
            early_stopping_var['cnt_wait_time'] = 0
            early_stopping_var['best_state_time'] = deepcopy(model.state_dict())
        else:
            early_stopping_var['cnt_wait_time'] += 1
        if new_best_mark and early_stopping_var['cnt_wait_mark'] < args.patience:
            early_stopping_var['best_loss_mark'] = val_metrics['loss_m']
            early_stopping_var['cnt_wait_mark'] = 0 
            early_stopping_var['best_state_mark'] = deepcopy(model.state_dict())
        else:
            early_stopping_var['cnt_wait_mark'] += 1
        if new_best_time or new_best_mark:
            early_stopping_var['best_state'] = deepcopy(model.state_dict())
        if early_stopping_var['cnt_wait_time'] > args.patience and early_stopping_var['cnt_wait_mark'] > args.patience:
            print(f'Early stopping! Stopping at epoch {epoch}')
            cnt_wait_time = early_stopping_var['cnt_wait_time']
            cnt_wait_mark = early_stopping_var['cnt_wait_mark']
            print(f'Time training stopped at epoch {epoch-cnt_wait_time}')
            print(f'Mark training stopped at epoch {epoch-cnt_wait_mark}')
            train_metrics_list.append({'stop epoch time':epoch-cnt_wait_time,
                                        'stop epoch mark':epoch-cnt_wait_mark})
            early_stopping_var['early_stop'] = True
        if early_stopping_var['cnt_wait_time'] > args.patience and early_stopping_var['cond_time']:
            for name, param in model.named_parameters():
                if 'mark' not in name:
                    param.requires_grad = False
            if early_stopping_var['cond_mark']:
                optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_rate_init)
            early_stopping_var['cond_time'] = False
        if early_stopping_var['cnt_wait_mark'] > args.patience and early_stopping_var['cond_mark']:
            for name, param in model.named_parameters():
                if 'mark' in name:
                    param.requires_grad = False
            if early_stopping_var['cond_time']:
                optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_rate_init)
            early_stopping_var['cond_mark'] = False
    return early_stopping_var, optimizer


def is_model_best(val_metrics, early_stopping_var, args):
    if not args.separate_training: 
        new_best_time, new_best_mark = True, True
        val_loss = val_metrics['loss_t'] + val_metrics['loss_m'] + val_metrics['loss_w']
        new_best = val_loss < early_stopping_var['best_loss']
        if args.loss_relative_tolerance is not None:
            abs_rel_loss_diff = (val_loss -early_stopping_var['best_loss']) /early_stopping_var['best_loss']
            abs_rel_loss_diff = abs(abs_rel_loss_diff)
            above_numerical_tolerance = (abs_rel_loss_diff >
                                            args.loss_relative_tolerance)
            new_best = new_best and above_numerical_tolerance
    else:
        val_loss_time = val_metrics['loss_t'] + val_metrics['loss_w']
        val_loss_mark = val_metrics['loss_m']
        new_best_time = val_loss_time < early_stopping_var['best_loss_time']
        new_best_mark = val_loss_mark < early_stopping_var['best_loss_mark']
        new_best = True
        if args.loss_relative_tolerance is not None:
            abs_rel_loss_diff_time = abs((val_loss_time - early_stopping_var['best_loss_time']) / early_stopping_var['best_loss_time'])
            abs_rel_loss_diff_mark = abs((val_loss_mark - early_stopping_var['best_loss_mark']) / early_stopping_var['best_loss_mark'])
            above_numerical_tolerance_time = (abs_rel_loss_diff_time >
                                            args.loss_relative_tolerance)
            above_numerical_tolerance_mark = (abs_rel_loss_diff_mark >
                                            args.loss_relative_tolerance)
            new_best_time = new_best_time and above_numerical_tolerance_time
            new_best_mark = new_best_mark and above_numerical_tolerance_mark    
    return new_best, new_best_time, new_best_mark

def set_model_state(args, model, early_stopping_var):
    if args.separate_training:
        model_dict = model.state_dict()
        mark_dict = {k: v for k, v in early_stopping_var['best_state_mark'].items() if 'mark' in k}
        model_dict.update(mark_dict)
        time_dict = {k: v for k, v in early_stopping_var['best_state_time'].items() if not 'mark' in k}
        model_dict.update(time_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(early_stopping_var['best_state'])
    return model

def save_model(model, args, exp_name):
    if args.save_check_dir is not None:
        file_path = f'{args.save_check_dir}/{exp_name}.pth'
        th.save(model.state_dict(), file_path)
        print('Model saved to disk.')

def save_args(args, exp_name):
    args_dic = vars(args).copy()
    args_dic.pop('device')
    for key, values in args_dic.items():
        if type(values) == np.ndarray:
            args_dic[key] = values.tolist()
    save_args_dir = f'{args.save_check_dir}/args'
    if not os.path.exists(save_args_dir):
        os.makedirs(save_args_dir)
    args_path = f'{save_args_dir}/{exp_name}.json'   
    with open(args_path , 'w') as f:
        json.dump(args_dic, f)
    print('Args saved.')

def get_exp_name(args):
    if args.encoder is not None:
        exp_name = args.encoder + '_' + args.decoder 
    elif args.encoder_histtime is not None:
        exp_name = args.encoder_histtime + '_' + args.encoder_histtime_encoding + '_' + args.encoder_histmark + '_' +  args.encoder_histmark_encoding + '_' +  args.decoder    
    if args.encoder_encoding is not None:
        exp_name += '_' + args.encoder_encoding
    if args.include_poisson:
        exp_name = args.dataset + '_poisson_' + exp_name
    else:
        exp_name = args.dataset + '_' +  exp_name
    if args.exp_name is not None:
        exp_name += '_' + args.exp_name
    if args.split is not None:
        exp_name = exp_name + '_split' + str(args.split)
    print(f'EXP NAME: {exp_name}')
    return exp_name

def save_results(train_metrics, val_metrics, test_metrics, 
                save_path, exp_name, args):
    results = {'train':train_metrics, 'val':val_metrics, 'test':test_metrics, 'args':args}
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = f'{save_path}/{exp_name}.txt'
    with open(save_path, "wb") as fp: 
        pkl.dump(results, fp)
    print(f'Results saved to {save_path}')

def get_state_dic(path:str):
    state_dic = th.load(path, map_location=th.device('cpu'))
    print(state_dic)

def get_early_stopping_var(args, model):
    var = {
        'best_loss': 1e9,
        'best_loss_time': 1e9,
        'best_loss_mark': 1e9,
        'cnt_wait': 0,
        'cnt_wait_time': 0,
        'cnt_wait_mark': 0,
        'early_stop': False, 
        'cond_time': True,
        'cond_mark': True
    }
    if args.separate_training:
        best_state = deepcopy(model.state_dict())
        best_state_time, best_state_mark = None, None  
    else:
        best_state = None 
        best_state_time = deepcopy(model.state_dict())
        best_state_mark = deepcopy(model.state_dict())
    var['best_state'] = best_state
    var['best_state_time'] = best_state_time
    var['best_state_mark'] = best_state_mark
    return var 
