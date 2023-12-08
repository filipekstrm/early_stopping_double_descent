import numpy as np
import torch



def linear_two_layer_simulation(Xs, ys, Xt, yt, Xs_low, U, ws, lr, args):

    
    # Some "pre-processing"
    
    # For the sake of not messing anything up
    Xs_t, ys_t, Xt_t, yt_t = Xs.T, ys.T, Xt.T, yt.T # TODO: transpose everything instead

    if args.transform_data:
        V = np.eye(args.dim)
        Uh = np.transpose(U)
        _, s, _ = np.linalg.svd(Xs_t.numpy(), full_matrices=True)
    else:
        V, s, Uh = np.linalg.svd(Xs_t.numpy(), full_matrices=True)

    V_tensor, Uh_tensor = torch.tensor(V, dtype=torch.float32), torch.tensor(Uh, dtype=torch.float32)
    S = torch.tensor(np.concatenate((s**2, np.zeros(args.dim - s.shape[0]))).reshape(1, -1), dtype=torch.float32)
    #St = ys_t @ Xs_t.T @ V_tensor

    beta_tensor = torch.tensor(beta, dtype=torch.float32).reshape(1, -1)
    eps_tensor = ((ys_t - beta_tensor @ Xs_t) @ Uh_tensor.T)[:, :args.dim]
    
    
    # Loss+risk function
    loss_fn = torch.nn.MSELoss(reduction='sum')
    risk_fn = torch.nn.L1Loss(reduction='mean') if args.risk_loss == 'L1' else loss_fn
    
    
    # For storing metrices
    losses = []
    losses_low = []
    losses_ind = []
    risks = []
    eigenvals = []
    weight_mse = []
    weights = []
    
    weights_norm = np.zeros((args.num_layers, int(args.iterations)))
    grad_norms = np.zeros((args.num_layers, int(args.iterations)))
    
    # Initialisation
    g_cpu = torch.Generator()
    g_cpu.manual_seed(args.seed)
    
    if args.u is None:
        u = torch.normal(0, torch.tensor(args.scales[1]), generator=g_cpu)
    else:
        u = torch.tensor(args.u)
    
    z = torch.normal(0, torch.tensor(args.scales[0]), size=(1, args.dim), generator=g_cpu)

    u_track, z_track = [], []
    u_track.append(u)
    z_track.append(z)
    
    # Initial evaluation    
    Wtot = u * z @ V_tensor.T
    y_pred = Wtot @ Xs_t

    loss = loss_fn(y_pred.T, ys_t.T)
    losses.append(loss.item())
 
    yt_pred = Wtot @ Xt_t

    risk = risk_fn(yt_pred.T, yt_t.T)
    risks.append(risk.item())
    
    if args.low_rank_eval:
        losses_low.append(np.array([loss_fn((Wtot @ Xs_l.T).T, ys_t.T).item() for Xs_l in Xs_low]))
    if args.ind_eval:
        losses_ind.append(np.array([loss_fn((Wtot @ Xs_t[i, :].reshape(-1, 1)).squeeze(), ys_t[i]).item() for i in range(args.samples)]))
    if args.weight_eval:
        assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
        weight_mse.append((Wtot.squeeze()-ws.squeeze())**2)
    if args.save_weights:
        weights.append(np.column_stack((u_track[-1], z_track[-1])))
    if args.eigen:
        print("Hessian eigenvalues not calculated for theoretical model") 
    

    # "Training"
    for t in range(int(args.iterations)):
        
        # Update parameters
        if args.u is None:
            grad_u = dudt_s(u_track[-1], z_track[-1], S, beta_tensor, eps_tensor) #dudt(u, z, S, St)
        else:
            grad_u = 0 # Keep u fixed
            
        grad_z = dzdt_s(u_track[-1], z_track[-1], S, beta_tensor, eps_tensor) #dzdt(u, z, S, St) 

        u = u + lr[1] * grad_u
        z = z + lr[0] * grad_z
        
        u_track.append(u)
        z_track.append(z)
        
        Wtot = u * z @ V_tensor.T
        
        weights_norm[0, i] = float(torch.norm(u))
        weights_norm[1, i] = float(torch.norm(z))

        grad_norms[0, i] = float(torch.norm(grad_u))
        grad_norms[1, i] = float(torch.norm(grad_z))


        # Evaluation
        y_pred = Wtot @ Xs_t

        loss = loss_fn(y_pred.T, ys_t.T)
        losses.append(loss.item())

        if not t % args.print_freq:
            print(t, loss.item())
            
        yt_pred = Wtot @ Xt_t

        risk = risk_fn(yt_pred.T, yt_t.T)
        risks.append(risk.item())
        
        if args.low_rank_eval:
            losses_low.append(np.array([loss_fn((Wtot @ Xs_l.T).T, ys_t.T).item() for Xs_l in Xs_low]))
        if args.ind_eval:
            losses_ind.append(np.array([loss_fn((Wtot @ Xs_t[i, :].reshape(-1, 1)).squeeze(), ys_t[i]).item() for i in range(args.samples)]))
        if args.weight_eval:
            assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
            weight_mse.append((Wtot.squeeze()-ws.squeeze())**2)
        if args.save_weights:
            weights.append(np.column_stack((u_track[-1], z_track[-1])))

        if not t % args.print_freq:
            print(t, risk.item())
            
            
    return {"loss": np.array(losses), "risk": np.array(risks), "weight_norm": weights_norm,
            "eigenvals": None, "grad_norm": grad_norms, "losslowrank": np.row_stack(losses_low) if losses_low else np.array(losses_low),
            "losses_ind": np.row_stack(losses_ind) if losses_low else np.array(losses_ind),
            "weight_mse": np.row_stack(weight_mse) if weight_mse else np.array(weight_mse), 
            "weights": np.row_stack(weights) if weights else np.array(weights)}



# With actual input data
def dt(u, z, S, St):
    assert S.shape == z.shape
    return (St - u * z * S)


def dzdt(u, z, S, St):
    return u * dt(u, z, S, St)


def dudt(u, z, S, St):
    return (dt(u, z, S, St) @ z.T).squeeze()


# Sampling only noise in output (and assuming that we know the true weights)
def dt_s(u, z, S, beta, eps):
    assert S.shape == z.shape
    return (beta - u * z) * S + eps * S**0.5


def dzdt_s(u, z, S, beta, eps):
    return u * dt_s(u, z, S, beta, eps)


def dudt_s(u, z, S, beta, eps):
    return (dt_s(u, z, S, beta, eps) @ z.T).squeeze()

