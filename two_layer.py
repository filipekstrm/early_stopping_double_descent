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
from linear_utils import linear_model
from train_utils import save_config

import wandb


def get_args():
    parser = argparse.ArgumentParser(description='CLI parameters for training')
    parser.add_argument('--config', type=str, default='', metavar='CONFIG',
                        help='Config file')
    parser.add_argument('--root', type=str, default='', metavar='DIR',
                        help='Root directory')
    parser.add_argument('-t', '--iterations', type=int, default=1e4, metavar='ITERATIONS',
                        help='Iterations (default: 1e4)')
    parser.add_argument('-n', '--samples', type=int, default=100, metavar='N',
                        help='Number of samples (default: 100)')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='CLI output printing frequency (default: 1000)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Number of GPUS to use')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('-d', '--dim', type=int, default=50, metavar='DIMENSION',
                        help='Feature dimension (default: 50)')
    parser.add_argument('--hidden', type=int, default=200, metavar='DIMENSION',
                        help='Hidden layer dimension (default: 200)')
    parser.add_argument('--batch-norm', action='store_true', default=False,
                        help='Use batch norm')
    parser.add_argument('--no-bias', action='store_true', default=False,
                        help='Do not use bias')
    parser.add_argument('--sigmas', type=str, default=None,
                        help='Sigmas')
    parser.add_argument('-r', '--s-range', nargs='*', type=float,
                        help='Range for sigmas')
    parser.add_argument('-w', '--scales', nargs='*', type=float,
                        help='scale of the weights')
    parser.add_argument('--lr', type=float, default=1e-4, nargs='*', metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--normalized', action='store_true', default=False,
                        help='normalize sample norm across features')
    parser.add_argument('--risk-loss', type=str, default='MSE', metavar='LOSS',
                        help='Loss for validation')
    parser.add_argument('--jacobian', action='store_true', default=False,
                        help='compute the SVD of the jacobian of the network')
    parser.add_argument('--save-results', action='store_true', default=False,
                        help='Save the results for plots')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot the results')
    parser.add_argument('--details', type=str, metavar='N',
                        default='no_detail_given',
                        help='details about the experimental setup')

    args = parser.parse_args(sys.argv[1:])

    # directories
    root = pathlib.Path(args.root) if args.root else pathlib.Path.cwd().parent

    current_date = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    args.outpath = (pathlib.Path.cwd().parent / 'results' / 'two_layer_nn' / current_date)

    if args.save_results:
        args.outpath.mkdir(exist_ok=True, parents=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args.device = 'cuda' if (not args.disable_cuda and torch.cuda.is_available()) else 'cpu'

    return args


def train_model(model, Xs, ys, Xt, yt, stepsize, args):
    # define loss functions
    loss_fn = torch.nn.MSELoss(reduction='sum')
    risk_fn = torch.nn.L1Loss(reduction='mean') if args.risk_loss == 'L1' else loss_fn
    losses = []
    risks = []
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
            i = -1
            for param in model.parameters():
                if len(param.shape) > 1:
                    i += 1
                param.data -= stepsize[i] * param.grad

        with torch.no_grad():
            model.eval()
            yt_pred = model(Xt)

            risk = risk_fn(yt_pred, yt)
            risks.append(risk.item())

            if not t % args.print_freq:
                print(t, risk.item())
            model.train()

    return np.array(losses), np.array(risks)


def init_model_params(model, args):
    # use kaiming initialization instead
    if args.scales:
        i = 0
        with torch.no_grad():
            for m in model:
                if type(m) == torch.nn.Linear:
                    if i == 0:
                        torch.nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                        m.weight.data = torch.mul(m.weight.data, args.scales[0])
                    if i == 1:
                        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                        m.weight.data = torch.mul(m.weight.data, args.scales[1])
                    print(m.weight.data.shape, args.scales[i])
                    i += 1
    return model


def get_model(args):
    model = torch.nn.Sequential(
        torch.nn.Linear(args.dim, args.hidden, bias=not args.no_bias),
        torch.nn.BatchNorm1d(args.hidden) if args.batch_norm else torch.nn.Identity(),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, 1, bias=not args.no_bias),
    ).to(args.device)
    model = init_model_params(model, args)
    return model


def get_dataset(args):
    # sample training set from the linear model
    lin_model = linear_model(args.dim, sigma_noise=0.0, normalized=False, sigmas=args.sigmas, s_range=args.s_range)
    Xs, ys = lin_model.sample(args.samples)
    Xs = torch.Tensor(Xs).to(args.device)
    ys = torch.Tensor(ys.reshape((-1, 1))).to(args.device)

    # sample the set for empirical risk calculation
    Xt, yt = lin_model.sample(args.samples)
    Xt = torch.Tensor(Xt).to(args.device)
    yt = torch.Tensor(yt.reshape((-1, 1))).to(args.device)
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


def append_id(filename, id):
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + [id])


def get_run_name(args):
    run_name = f'lr={args.lr[0]}_{args.lr[1]}'
    if args.batch_norm:
        run_name += "_batch_norm"
    return run_name


def get_result_path(args):
    run_name = get_run_name(args)
    result_path = os.path.join("two_layer_results", run_name + ".csv")
    return result_path


def save_results(risks, args):
    result_path = get_result_path(args)
    data = pd.DataFrame(risks)
    data.to_csv(result_path, header=False, index=False)


def plot_results_from_file(result_path):
    """
    For debugging
    """
    data = pd.read_csv(result_path, header=None)
    data.plot()
    plt.show()


def main(args):
    wandb.init(project="double_descent", name=get_run_name(args), config=args)
    Xs, ys, Xt, yt = get_dataset(args)
    model_diff = get_model(args)
    loss, risks = train_model(model_diff, Xs, ys, Xt, yt, args.lr, args)
    save_results(risks, args)

    if args.plot:  # for debugging
        plot_results_from_file(os.path.join('two_layer_runs', f'lr={args.lr[0]}_{args.lr[1]}.csv'))


if __name__ == "__main__":
    args = get_args()
    main(args)