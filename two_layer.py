import matplotlib
import matplotlib.pyplot as plt

import argparse
import os
import datetime
import pathlib
import random
import json
import numpy as np
import math
import pandas as pd

import torch

import sys
sys.path.append('code/')
from linear_utils import linear_model, is_float
from train_utils import save_config, prune_data, calculate_weight_mse, extract_weights, ScalingLayer, kaiming_init, fixed_init, rank_one_init
from theoretical_model import linear_two_layer_simulation

from sharpness_utilities import sharpness

import wandb


def train_model(model, Xs, ys, Xt, yt, Xs_low, true_weights, stepsize, args):
    # define loss functions
    
        
    loss_fn = torch.nn.MSELoss(reduction='sum')
    risk_fn = torch.nn.L1Loss(reduction='mean') if args.risk_loss == 'L1' else loss_fn
    
    losses = []
    losses_low = []
    losses_ind = []
    risks = []
    eigenvals = []
    weight_mse = []
    weight_mse_min = []
    weights = []
    
    weights_norm = np.zeros((args.num_layers, int(args.iterations)))
    grad_norms = np.zeros((args.num_layers, int(args.iterations)))
    
    w_min = 0
    if args.linear and args.dim < args.samples and args.no_bias:
            w_min = np.linalg.solve(np.transpose(Xs)@Xs, np.transpose(Xs)@ys)
            loss_min = loss_fn(Xs@w_min, ys)
            print(f"Minimum loss: {loss_min}")
            
        
    if args.num_layers == 1:
    
        epoch_fun = train_epoch_one_layer
    
        if np.isclose(stepsize[0], stepsize[1], rtol=1e-10):
            stepsize = stepsize[0]
        else:
            if args.p is None:
                stepsize = torch.tensor(([stepsize[0]] * int(np.ceil(args.dim / 2)), [stepsize[1]] * int(np.floor(args.dim / 2)))).reshape(-1)
            else:
                stepsize = torch.tensor(([stepsize[0]] * p, [stepsize[1]] * (args.dim - p))).reshape(-1)
            
            print(stepsize)
        
    else:
        epoch_fun = train_epoch
        
    if args.eig_val_frac is None:
        p = int(args.dim / 2)
    else:
        p = int(args.eig_val_frac)
        
       
    # Store risk (and eigenvalues) at initialisation as well
    model.eval()
    with torch.no_grad():
        risks.append(risk_fn(model(Xt), yt).item())
        
        if args.low_rank_eval:
            losses_low.append(np.array([loss_fn(model(Xs_l), ys).item() for Xs_l in Xs_low]))
        if args.ind_eval:
            losses_ind.append(np.array([loss_fn(model(Xs[i, :].reshape(1, -1)), ys[i]).item() for i in range(args.samples)]))
        if args.weight_eval:
            assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
            weight_mse.append(calculate_weight_mse(model, true_weights, p=p))
        if args.weight_eval_min:
            assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
            weight_mse_min.append(calculate_weight_mse(model, w_min.squeeze(), p=p))
        if args.save_weights:
            weights.append(extract_weights(model))
          

    if args.eigen:
        evals = sharpness.get_hessian_eigenvalues(model, loss_fn, sharpness.DatasetWrapper(Xs, ys), args)
        eigenvals.append(float(evals[0]))
        
    model.train()
    for t in range(int(args.iterations)):
        y_pred = model(Xs)

        loss = loss_fn(y_pred, ys)
        losses.append(loss.item())

        if not t % args.print_freq:
            print(t, loss.item())

        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            weights_norm[:, t], grad_norms[:, t] = epoch_fun(model, stepsize, args)

        model.eval()
        with torch.no_grad():
            yt_pred = model(Xt)

            risk = risk_fn(yt_pred, yt)
            risks.append(risk.item())

            if not t % args.print_freq:
                print(t, risk.item())
                
            if args.low_rank_eval:
                losses_low.append(np.array([loss_fn(model(Xs_l), ys).item() for Xs_l in Xs_low]))
                
            if args.ind_eval:
                losses_ind.append(np.array([loss_fn(model(Xs[i, :].reshape(1, -1)), ys[i]).item() for i in range(args.samples)]))

            if args.weight_eval:
                assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
                weight_mse.append(calculate_weight_mse(model, true_weights, p=p))
                
            if args.weight_eval_min:
                assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
                weight_mse_min.append(calculate_weight_mse(model, w_min.squeeze(), p=p))
                       
            if args.save_weights:
                weights.append(extract_weights(model))
        
        
        if args.eigen:
            evals = sharpness.get_hessian_eigenvalues(model, loss_fn, sharpness.DatasetWrapper(Xs, ys), args)
            eigenvals.append(float(evals[0]))
        
        model.train()
        
    
    y_pred = model(Xs)
    losses.append(loss_fn(y_pred, ys).item())

    return {"loss": np.array(losses), "risk": np.array(risks), "weight_norm": weights_norm,
            "eigenvals": np.array(eigenvals), "grad_norm": grad_norms, "losslowrank": np.row_stack(losses_low) if losses_low else np.array(losses_low),
            "losses_ind": np.row_stack(losses_ind) if losses_low else np.array(losses_ind),
            "weight_mse": np.row_stack(weight_mse) if weight_mse else np.array(weight_mse), 
            "weight_mse_min": np.row_stack(weight_mse_min) if weight_mse_min else np.array(weight_mse_min),
            "weights": np.row_stack(weights) if weights else np.array(weights)}


def train_epoch(model, stepsize, args):
    
    weights_norm = np.zeros((args.num_layers))
    grad_norms = np.zeros((args.num_layers))
    
    i = -1
    for param in model.parameters():
        
        if param.grad is not None:
            if len(param.shape) > 1:
                i += 1
                weights_norm[i] = float(torch.norm(param.data.flatten()))
                grad_norms[i] = float(torch.norm(param.grad.flatten()))
            #param.data -= stepsize[i] * param.grad
            
            if i < (args.num_layers - 1):
                param.data -= stepsize[0] * param.grad
                
                #if t == 0:
                #    print(f'I did pass lr {stepsize[0]}')
            else:
                assert i == (args.num_layers - 1), "Something is wrong with the amount of layers"
                                
                param.data -= stepsize[1] * param.grad
                
                #if t == 0:
                #    print(f'Using lr {stepsize[1]}')
                
    return weights_norm, grad_norms

def train_epoch_one_layer(model, stepsize, args):
    
    weights_norm = np.zeros((args.num_layers))
    grad_norms = np.zeros((args.num_layers))
    
    i = -1
    for param in model.parameters():
        if len(param.shape) > 1:
            i += 1
            weights_norm[i] = float(torch.norm(param.data.flatten()))
            grad_norms[i] = float(torch.norm(param.grad.flatten()))

        param.data -= stepsize * param.grad
                
    return weights_norm, grad_norms

def init_model_params(model, args):
    # use kaiming initialization instead
    
    if args.scales:

        g_cpu = torch.Generator()
        g_cpu.manual_seed(args.seed)

        if args.fixed_weight_init:
            model = fixed_init(model, args)
        elif args.one_rank_init:
            model = rank_one_init(model, g_cpu, args)
        else:
            model = kaiming_init(model, g_cpu, args)
        
    else:
        print("Default initialisation, no scales given.")

    return model


def get_model(args):

    model_dict = {1: one_layer_model, 2: two_layer_model, 5: five_layer_model}
        
    if args.num_layers in model_dict:
        model = model_dict[args.num_layers](args)
    else:
        print("Model not available, using 2 layer model")
        model = two_layer_model(args)
    
    model = init_model_params(model, args)
    return model

def one_layer_model(args):
    model = torch.nn.Sequential(
        torch.nn.Linear(args.dim, 1, bias=not args.no_bias),
    ).to(args.device)
    
    return model 

    
def two_layer_model(args):

    if args.scaling_layer:
        model = torch.nn.Sequential(
            ScalingLayer(args.dim, args.hidden)
            ).to(args.device)
    
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(args.dim, args.hidden, bias=not args.no_bias),
            torch.nn.BatchNorm1d(args.hidden) if args.batch_norm else torch.nn.Identity(),
            torch.nn.Identity() if args.linear else torch.nn.ReLU(),
            torch.nn.Linear(args.hidden, 1, bias=not args.no_bias),
        ).to(args.device)
    
    return model
    
def five_layer_model(args):
    model = torch.nn.Sequential(
        torch.nn.Linear(args.dim, args.hidden, bias=not args.no_bias),
        torch.nn.BatchNorm1d(args.hidden) if args.batch_norm else torch.nn.Identity(),
        torch.nn.Identity() if args.linear else torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.hidden, bias=not args.no_bias),
        torch.nn.BatchNorm1d(args.hidden) if args.batch_norm else torch.nn.Identity(),
        torch.nn.Identity() if args.linear else torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.hidden, bias=not args.no_bias),
        torch.nn.BatchNorm1d(args.hidden) if args.batch_norm else torch.nn.Identity(),
        torch.nn.Identity() if args.linear else torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, int(args.hidden / 2), bias=not args.no_bias),
        torch.nn.BatchNorm1d(args.hidden) if args.batch_norm else torch.nn.Identity(),
        torch.nn.Identity() if args.linear else torch.nn.ReLU(),
        torch.nn.Linear(int(args.hidden / 2), 1, bias=not args.no_bias),
    ).to(args.device)
    
    return model
    
def freeze_layer(model, args):
            
    assert args.num_layers >= args.freeze_layer, "Layer to be freezed does not exist"

    count = 1
    for param in model.parameters():
        if len(param.shape) > 1:
            if count == args.freeze_layer:
                param.requires_grad = False
            else:
                count += 1
    
    return model

def get_dataset(args, return_extra=False):
    # sample training set from the linear model
    
    if args.beta is not None:
        args.beta = np.array(args.beta)           
        
    if args.eig_val_frac is None:
        p = None
    else:
        p = args.eig_val_frac*args.dim
  
    lin_model = linear_model(args.dim, sigma_noise=args.sigma_noise, beta=args.beta, scale_beta=args.scale_beta, normalized=False, sigmas=args.sigmas, s_range=args.s_range, coupled_noise=args.coupled_noise, transform_data=args.transform_data, kappa=args.kappa, p=p)
    Xs, ys = lin_model.sample(args.samples, train=True)
    Xs = torch.Tensor(Xs).to(args.device)
    ys = torch.Tensor(ys.reshape((-1, 1))).to(args.device)

    # sample the set for empirical risk calculation
    Xt, yt = lin_model.sample(args.samples, train=False) # * 1000
    Xt = torch.Tensor(Xt).to(args.device)
    yt = torch.Tensor(yt.reshape((-1, 1))).to(args.device)
    
    if return_extra:
        return Xs, ys, Xt, yt, lin_model.right_singular_vecs, lin_model.beta
    else:
        return Xs, ys, Xt, yt


def plot_results(risks, risks_same, args):
    geo_samples = [int(i) for i in np.geomspace(1, len(risks) - 1, num=700)]
    cmap = matplotlib.cm.get_cmap('viridis')
    colorList = [cmap(50 / 1000), cmap(350 / 1000)]
    labelList = ['same stepsize', 'scaled stepsize']

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    ax.set_xscale('log')

    ax.plot(geo_samples, risks_same[geo_samples],
            color=colorList[0],
            label=labelList[0],
            lw=4)
    ax.plot(geo_samples, risks[geo_samples],
            color=colorList[1],
            label=labelList[1],
            lw=4)

    ax.legend(loc=1, bbox_to_anchor=(1, 1), fontsize='x-large',
              frameon=False, fancybox=True, shadow=True, ncol=1)
    ax.set_ylabel('risk')
    ax.set_xlabel(r'$t$ iterations')
    ax.set_title(f'Initialization scale = {args.scales}, Learning rates = {args.lr}')

    plt.show()


def plot_individual_run(risks, args, label, figname=None):
    geo_samples = [int(i) for i in np.geomspace(1, len(risks) - 1, num=700)]
    cmap = matplotlib.colormaps['viridis']
    colorList = [cmap(50 / 1000), cmap(350 / 1000)]

    #fig = plt.figure()
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.plot(geo_samples, risks[geo_samples],
        color=colorList[1],
        label=label,
        lw=4)
    plt.legend()

    plt.title(fr"$\eta_{{\mathbf{{W}}}} = {args.lr[0]}$, $\eta_{{\mathbf{{v}}}} = {args.lr[1]}$")
    if figname is not None:
        plt.savefig(os.path.join("plots", figname), bbox_inches=None)
    plt.show()


def append_id(filename, id):
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + [id])


def get_run_name(args):

    key_word_dict = {"lr1": args.lr[0], "lr2": args.lr[1], "kappa": args.kappa, "hidden": args.hidden, "": ""}
            
    if args.key_word in key_word_dict:
        run_name = args.key_word + f"_{key_word_dict[args.key_word]}"
    else:
        print("Unrecognised key word")
        run_name = args.key_word
        
        assert args.key_word != "", "Empty run name"
        
    return run_name


def get_result_path(args):
    run_name = get_run_name(args)
    
    layer_dir_dict = {1: "one_layer", 2: "two_layer", 5: "five_layer_regression"}
    risk_dir_dict = {'L1': "", 'L2': "_l2"}
    
    if args.num_layers in layer_dir_dict:
        base_dir = "results/" + layer_dir_dict[args.num_layers] + "_results"
    else:
        print("Model not available, assuming two layer model")
        args.num_layers = 2
        base_dir = "results/" + layer_dir_dict[args.num_layers] + "_results"
        
    base_dir += risk_dir_dict[args.risk_loss]
        
    if args.transform_data:
        base_dir = os.path.join(base_dir, "transform_data")
        
    if args.freeze_layer:
        base_dir = os.path.join(base_dir, "layer_freeze_" + str(args.freeze_layer))
        
    if args.scaling_layer:
        base_dir = os.path.join(base_dir, "scaling_layer")
        
    if args.theoretical:
        base_dir = os.path.join(base_dir, "theoretical")
        
    if args.sweep is not None:
        base_dir = os.path.join(base_dir, args.sweep)
        
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # (io.get_checkpoint_root())
    
    return base_dir


def save_results(args, res):
    result_path = get_result_path(args)
    run_name = get_run_name(args)
    
    result_file = os.path.join(result_path, run_name + ".txt")
    print("Saving data to " + result_file)
    
    unravelled_res = {}
    for key, value in res.items():
        if value.ndim > 1:
            for i in range(value.shape[-1]):
                unravelled_res[key + "_" + str(i + 1)] = value[:, i].tolist()
            
        else:
            unravelled_res[key] = value.tolist()
    
    with open(result_file, "w") as f:
        json.dump(unravelled_res, f)
        

def plot_results_from_file(result_path, args):
    """
    For debugging
    """
    data = pd.read_csv(result_path, header=None)
    plot_individual_run(data[0], args, "risk")
    plt.show()


def main(args):
    wandb.init(project="double_descent", name=get_run_name(args), config=args)
    
    Xs, ys, Xt, yt, U, ws = get_dataset(args, return_extra=True)
    
    if args.pcs is not None:
        Xs = prune_data(Xs, args.pcs)
            
    Xs_low = None
    if args.low_rank_eval:
        Xs_low = [prune_data(Xs, int(i)) for i in np.arange(10, args.dim, 10)]

    if args.theoretical:
        assert args.num_layers == 2 and args.linear, "Theoretical simulation only for linear two layer model"
                
        out = linear_two_layer_simulation(Xs, ys, Xt, yt, Xs_low, U, ws, args.lr, args)
        
    else:
    
        model = get_model(args)
        
        if args.freeze_layer:
            model = freeze_layer(model, args)
        print(model)
               
        out = train_model(model, Xs, ys, Xt, yt, Xs_low, ws, args.lr, args) 

    #additional_data = [out[key] for key in out if (out[key].size != 0 and key not in ["risk", "loss", "weight_norm", "grad_norm"])] 
    #save_results(args, out["risk"], out["loss"], additional_data)
    save_results(args, out)
    
    if args.plot:  # for debugging
        plot_results_from_file(get_result_path(args), args)
        plot_individual_run(out["risk"], args, "Risk", append_id(get_run_name(args) + ".pdf", "risk"))
        plot_individual_run(out["weight_norm"][0], args, "W norm", append_id(get_run_name(args) + ".pdf", "W_norm"))
        plot_individual_run(out["weight_norm"][1], args, "v norm", append_id(get_run_name(args) + ".pdf", "v_norm"))
        if args.eigen:
            plot_individual_run(out["eigenvals"], args, "Leading eigenvalue of Hessian", append_id(get_run_name(args) + ".pdf", "eigenvals"))
        plot_individual_run(out["grad_norm"][0]*args.lr[0], args, " effective grad(W) norm", append_id(get_run_name(args) + ".pdf", "w_grad_norm"))
        plot_individual_run(out["grad_norm"][1]*args.lr[1], args, "effective grad(v) norm",
                            append_id(get_run_name(args) + ".pdf", "v_grad_norm"))

if __name__ == "__main__":
    args = get_args()
    main(args)