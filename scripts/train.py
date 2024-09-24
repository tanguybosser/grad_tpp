import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'neurips_supplementary')))

import json
import numpy as np
import os
import time
import sys
import pickle as pkl
from collections import defaultdict

import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset


from argparse import Namespace
from copy import deepcopy
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from tpps.utils.events import get_events, get_window

import scripts.utils as utils 
from tpps.models import get_model
from tpps.models.base.process import Process
from tpps.utils.cli import parse_args
from tpps.utils.metrics import mark_metrics, mae_time_prediction, get_time_prediction
from tpps.utils.data import get_loader, load_data
from tpps.utils.run import make_deterministic
from tpps.utils.stability import check_tensor

print('cuda', th.cuda.is_available(), flush=True)
sys.dont_write_bytecode = True

def get_loss(
        model: Process,
        batch: Dict[str, th.Tensor],
        args: Namespace,
        eval_metrics: Optional[bool] = False,
        test: Optional[bool] = False
) -> Tuple[th.Tensor, th.Tensor, Dict]:
    """Compute the loss (NLL) and predict marks.

    Args:
        model: The model for which the loss is computed.
        batch: batch of sequences.
        args: arguments.
        eval_metrics: If True, the events mark are predicted as the highest probability. 
        test: If True, compute CDF of arrival times. 
    Returns:
        Time, mark and window losses, along auxilary quantities in artifact. 

    """
    times, labels = batch["times"], batch["labels"]
    labels = (labels != 0).type(labels.dtype) 
    mask = (times != args.padding_id).type(times.dtype)
    times = times * args.time_scale 
    window_start, window_end = get_window(times=times, window=args.window) 
    events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
    loss_t, loss_m, loss_w, loss_mask, artifacts = model.neg_log_likelihood(events=events, test=test, time_scaling=args.nll_scaling)  # [B]
    if eval_metrics:
        events_times = events.get_times()
        log_p, log_mark_density, y_pred_mask = model.log_density(
        query=events_times, events=events)
        proba = th.exp(log_mark_density)
        if args.multi_labels:
            y_pred = log_p  # [B,L,M]
            labels = events.labels
        else:
            y_pred = log_mark_density.argmax(-1).type(events.labels.dtype)  # [B,L]
            labels = events.labels.argmax(-1).type(events.labels.dtype)
            max_proba  = proba.max(-1).values.type(log_mark_density.dtype)
            proba = proba.detach().cpu().numpy()
            ranks = (-proba).argsort(-1).argsort(-1)
            ranks_idx = labels.unsqueeze(-1) #[B,L,1]
            ranks_idx = ranks_idx.detach().cpu().numpy().astype(np.int32)            
            ranks_proba = np.take_along_axis(ranks, ranks_idx, axis=-1).squeeze(-1) #[B,L]
            artifacts['ranks proba'] = ranks_proba
        artifacts['y_pred'] = y_pred
        artifacts['y_true'] = labels
        artifacts['y_pred_mask'] = y_pred_mask
        artifacts['max proba'] = max_proba
    return loss_t, loss_m, loss_w, loss_mask, artifacts


def evaluate(model: Process, args: Namespace, loader: DataLoader, test: Optional[bool] = False, eval_metrics: Optional[bool] =False
             ) -> Dict[str, float]:
    """Evaluate a model on a specific dataset.

    Args:
        model: The model to evaluate.
        args: Arguments for evaluation
        loader: The loader corresponding to the dataset to evaluate on.
        test: If False, put what is not returned by log-likelihhod. 
        eval_metrics: If True, compute evaluation metrics. 
    Returns:
        Dictionary of evaluation metrics. 
    
    """
    model.eval()
    t0, epoch_loss, epoch_loss_per_time, n_seqs = time.time(), 0., 0., 0.
    epoch_loss_t, epoch_loss_m, epoch_loss_w = 0, 0, 0 
    pred_labels, gold_labels, mask_labels, probas, ranks, target_times, predicted_times = [], [], [], [], [], [], []
    results = {}
    epoch_h_t, epoch_h_m, epoch_h = [], [], []
    cumulative_density = []
    num_batch = 0
    for batch in tqdm(loader) if args.verbose else loader:
        batch['times'], batch['labels'], batch['seq_lens'] = batch['times'].to(args.device), batch['labels'].to(args.device), batch['seq_lens'].to(args.device)
        loss_t, loss_m, loss_w, loss_mask, artifacts = get_loss(  # [B]
            model, batch=batch, eval_metrics=eval_metrics, 
            args=args, test=test) 
        loss_t = loss_t * loss_mask #[B]
        loss_m = loss_m * loss_mask
        loss_w = loss_w * loss_mask

        true_loss_t = artifacts['loss_t'] * loss_mask
        true_loss_m = artifacts['loss_m'] * loss_mask      
        true_loss_w = artifacts['loss_w'] * loss_mask

        epoch_loss_t += utils.detach(th.sum(true_loss_t))
        epoch_loss_m += utils.detach(th.sum(true_loss_m))
        epoch_loss_w += utils.detach(th.sum(true_loss_w))
        epoch_loss += epoch_loss_t + epoch_loss_m + epoch_loss_w
        
        if test:
            cdf = artifacts['cumulative density'].cpu().numpy()
            valid_cdf = [cdf[i][cdf[i] >=0].tolist() for i in range(cdf.shape[0])]
            cumulative_density.extend(valid_cdf)
        if test and 'last_h_t' in artifacts:
            epoch_h_t.append(artifacts['last_h_t'])
            epoch_h_m.append(artifacts['last_h_m'])
        if test and 'last_h' in artifacts:
            epoch_h.append(artifacts['last_h'])
        n_seqs_batch = utils.detach(th.sum(loss_mask))
        n_seqs += n_seqs_batch
        num_batch += 1
        if eval_metrics:
            pred_labels.append(utils.detach(artifacts['y_pred']))
            gold_labels.append(utils.detach(artifacts['y_true']))
            mask_labels.append(utils.detach(artifacts['y_pred_mask']))
            probas.append(utils.detach(artifacts['max proba']))
            ranks.append(artifacts['ranks proba'])
    if eval_metrics: 
        results = mark_metrics(
            pred=pred_labels,
            gold=gold_labels,
            probas=probas,
            ranks=ranks,
            mask=mask_labels,
            results=results,
            n_class=args.marks,
            multi_labels=args.multi_labels,
            test=test)
        predictions, target = get_time_prediction(model, batch, args)
        predicted_times.append(utils.detach(predictions))
        target_times.append(utils.detach(target))
        mae = mae_time_prediction(predicted_times, target_times)
        results["mae"] = mae
    dur = time.time() - t0
    results["dur"] = dur
    results["loss"] = float(epoch_loss / n_seqs)
    results['loss_t'] = float(epoch_loss_t / n_seqs)
    results['loss_m'] = float(epoch_loss_m / n_seqs)
    results['loss_w'] = float(epoch_loss_w / n_seqs)
    if test:
        results["cdf"] = cumulative_density
        if 'last_h_t' in artifacts:
            last_h_t = np.concatenate(epoch_h_t)
            last_h_m = np.concatenate(epoch_h_m)
            results['last_h_t'] = last_h_t
            results['last_h_m'] = last_h_m
        if 'last_h' in artifacts:
            last_h = np.concatenate(epoch_h)
            results['last_h'] = last_h
    return results


def train(
        model: Process,
        args: Namespace,
        loader: DataLoader,
        val_loader: DataLoader) -> Tuple[Process, dict, list, list]:
    """Train a model.

    Args:
        model: Model to be trained.
        args: Arguments for training.
        loader: The dataset for training.
        val_loader: The dataset for evaluation.
    Returns:
        Best trained model from early stopping.

    """
    print("Training starts.")
    train_metrics_list, val_metrics_list = [], []
    optimizer = Adam(model.parameters(), lr=args.lr_rate_init)
    early_stopping_var = utils.get_early_stopping_var(args, model)
    val_dur = list()
    epochs = range(args.train_epochs)
    if args.verbose:
        epochs = tqdm(epochs)
    t_start = time.time()
    epoch_dot_products = {k:[] for k,_ in model.named_parameters()}
    epoch_grad_sim = {k:[] for k,_ in model.named_parameters()}
    epoch_grad_tpi = {k:[] for k,_ in model.named_parameters()}
    for j, epoch in enumerate(epochs):
        t0, _ = time.time(), model.train()
        train_metrics = {}
        epoch_loss_time, epoch_loss_mark, epoch_loss_window, epoch_loss, n_seqs = 0, 0, 0, 0, 0
        time_grads, mark_grads = defaultdict(), defaultdict()
        dot_products = {k:[] for k,_ in model.named_parameters()}
        grad_sim = {k:[] for k,_ in model.named_parameters()}
        grad_tpi = {k:[] for k,_ in model.named_parameters()}
        for i, batch in enumerate((tqdm(loader)) if args.verbose else loader):
            batch['times'], batch['labels'], batch['seq_lens'] = batch['times'].to(args.device), batch['labels'].to(args.device), batch['seq_lens'].to(args.device)
            optimizer.zero_grad()
            loss_t, loss_m, loss_w, loss_mask, artifacts = get_loss(model, batch=batch, args=args, test=False)  # [B]
            loss_t = th.sum(loss_t * loss_mask)
            loss_mark = th.sum(loss_m * loss_mask)
            loss_w = th.sum(loss_w * loss_mask)
            loss_time = loss_t + loss_w 
            check_tensor(loss_time)
            check_tensor(loss_mark)
            loss_time.backward(retain_graph=True)
            for name, p in model.named_parameters():
                if p.requires_grad:            
                    if p.grad is None:
                        grad = th.zeros_like(p.data, device='cpu').detach().view(-1)
                    else:
                        grad = p.grad.data.detach().clone().cpu().view(-1)
                    time_grads[name] = grad
            loss_mark.backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    #Grads are accumulated, so we must subtract the time grads to get the mark grads. 
                    mark_grads[name] = p.grad.data.detach().clone().cpu().view(-1) - time_grads[name]            
            dot_products, grad_sim, grad_tpi = utils.compute_grad_metrics(time_grads, mark_grads, dot_products,
                                                                    grad_sim, grad_tpi) 
            optimizer.step()
            epoch_loss_time += utils.detach(loss_time)
            epoch_loss_mark += utils.detach(loss_mark)
            epoch_loss_window += utils.detach(loss_w)
            epoch_loss += epoch_loss_time + epoch_loss_mark + epoch_loss_window
            n_seqs += utils.detach(th.sum(loss_mask))
        train_metrics['dur'] = time.time() - t0
        train_metrics['loss'] = float(epoch_loss/n_seqs)
        train_metrics['loss_t'] = float(epoch_loss_time/n_seqs)
        train_metrics['loss_m'] = float(epoch_loss_mark/n_seqs)
        train_metrics['loss_w'] = float(epoch_loss_window/n_seqs)
        train_metrics_list.append(train_metrics)
        val_metrics = evaluate(model, args=args, loader=val_loader, test=False, eval_metrics=False)
        val_dur.append(val_metrics["dur"])
        val_metrics_list.append(val_metrics)
        for k, v in dot_products.items():
            epoch_dot_products[k].extend(v) 
            epoch_grad_sim[k].extend(grad_sim[k])
            epoch_grad_tpi[k].extend(grad_tpi[k])
        #EARLY STOPPING 
        early_stopping_var, optimizer = utils.early_stopping(args, val_metrics, train_metrics_list,
                                                    early_stopping_var, 
                                                    model, epoch, optimizer)      
        if early_stopping_var['early_stop']:
            break
    train_metrics_list.append({'dot_products':epoch_dot_products, 'grad_sim':epoch_grad_sim, 'grad_ind': epoch_grad_tpi})
    model = utils.set_model_state(args, model, early_stopping_var)
    delta_t = time.time() - t_start
    hours, mins, sec = int(delta_t/3600), int((delta_t%3600)/60), int((delta_t%3600)%60)
    print(f'Total training time : {hours}:{mins}:{sec}')
    return model, train_metrics_list, val_metrics_list
    
def main(args: Namespace):
    datasets = load_data(args=args) 
    loaders = dict()
    loaders["train"] = get_loader(datasets["train"], args=args, shuffle=True)
    loaders["val"] = get_loader(datasets["val"], args=args, shuffle=False)
    loaders["test"] = get_loader(datasets["test"], args=args, shuffle=False)
    exp_name = utils.get_exp_name(args)
    utils.save_args(args, exp_name)
    model = get_model(args)
    utils.count_parameters(model)
    model, train_metrics, val_metrics = train(
        model, args=args, 
        loader=loaders["train"],
        val_loader=loaders["val"]) 
    print('Training complete.')
    utils.save_model(model, args, exp_name)
    print("Evaluating model.")
    t_val = time.time()
    metrics = {
    'train':evaluate(model=model, args=args, loader=loaders["train"], test=True, eval_metrics=False),
    'val':evaluate(model=model, args=args, loader=loaders["val"], test=True, eval_metrics=False),
    'test':evaluate(model=model, args=args, loader=loaders["test"], test=True, eval_metrics=args.eval_metrics)
    }
    delta_t = time.time() - t_val
    hours, mins, sec = int(delta_t/3600), int((delta_t%3600)/60), int((delta_t%3600)%60)
    print(f'Total validation time : {hours}:{mins}:{sec}')
    train_metrics.append(metrics['train']) 
    val_metrics.append(metrics['val'])
    test_metrics = metrics['test']
    if args.verbose:
        print(metrics, flush=True)
    if args.save_results_dir is not None:
        utils.save_results(train_metrics, val_metrics, test_metrics,save_path=args.save_results_dir, exp_name=exp_name ,args=args)


if __name__ == "__main__":
    parsed_args = parse_args()
    json_dir = f'{os.getcwd()}/{parsed_args.load_from_dir}/{parsed_args.dataset}'
    if parsed_args.split is not None:
        json_dir = f'{json_dir}/split_{parsed_args.split}'
    json_path = f'{json_dir}/args.json'
    with open(json_path, 'r') as fp:
        args_dict_json = json.load(fp)
    args_dict = vars(parsed_args)
    shared_keys = set(args_dict_json).intersection(set(args_dict))
    for k in shared_keys:
        v1, v2 = args_dict[k], args_dict_json[k]
        is_equal = np.allclose(v1, v2) if isinstance(
            v1, np.ndarray) else v1 == v2
        if not is_equal:
            print(f"    {k}: {v1} -> {v2}", flush=True)
    args_dict.update(args_dict_json)
    parsed_args = Namespace(**args_dict)
    cuda = th.cuda.is_available() and not parsed_args.disable_cuda
    if cuda:
        parsed_args.device = th.device('cuda')
    else:
        parsed_args.device = th.device('cpu')        
    if not parsed_args.include_window:
        parsed_args.window = None  
    make_deterministic(seed=parsed_args.seed)
    main(args=parsed_args)
