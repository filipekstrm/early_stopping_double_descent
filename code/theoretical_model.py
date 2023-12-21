import numpy as np
import torch



def linear_two_layer_simulation(Xs, ys, Xt, yt, Xs_low, U, ws, lr, args):

    
    # Some "pre-processing"
    if args.transform_data:
        V = np.eye(args.dim)
        _, s, _ = np.linalg.svd(Xs.numpy(), full_matrices=True)
    else:
        Uh, s, V = np.linalg.svd(Xs.numpy(), full_matrices=True)
        U = np.transpose(Uh)

    V_tensor, U_tensor = torch.tensor(V, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
    S = torch.tensor(np.concatenate((s**2, np.zeros(args.dim - s.shape[0]))).reshape(1, -1), dtype=torch.float32)
    #St = ys_t @ Xs_t.T @ V_tensor

    beta_tensor = torch.tensor(ws, dtype=torch.float32).reshape(1, -1)
    eps_tensor = ((ys - Xs @ beta_tensor.T) @ U_tensor)[:, :args.dim]
    
    assert ys.shape == (Xs.shape[0], 1)
    
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
    weight_mse_min = []
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
    y_pred = Xs @ Wtot.T

    loss = loss_fn(y_pred, ys)
    losses.append(loss.item())
 
    yt_pred = Xt @ Wtot.T

    risk = risk_fn(yt_pred, yt)
    risks.append(risk.item())
    
    
    w_min = 0
    if args.linear and args.dim < args.samples and args.no_bias:
        w_min = np.linalg.solve(np.transpose(Xs)@Xs, np.transpose(Xs)@ys)
        loss_min = loss_fn(Xs@w_min, ys)
        print(f"Minimum loss: {loss_min}")
    
    if args.low_rank_eval:
        losses_low.append(np.array([loss_fn((Xs_l @ Wtot.T, ys)).item() for Xs_l in Xs_low]))
    if args.ind_eval:
        losses_ind.append(np.array([loss_fn((Xs[i, :].reshape(1, -1) @ Wtot.T).squeeze(), ys[i]).item() for i in range(args.samples)]))
    if args.weight_eval:
        assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
        weight_mse.append((Wtot.squeeze()-ws.squeeze())**2)
    if args.weight_eval_min:
        assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
        weight_mse_min.append((Wtot.squeeze()-w_min.squeeze())**2)
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
        
        if args.u is None:
            weights_norm[0, t] = float(torch.norm(u))
            grad_norms[0, t] = float(torch.norm(grad_u))

        weights_norm[1, t] = float(torch.norm(z))
        grad_norms[1, t] = float(torch.norm(grad_z))


        # Evaluation
        y_pred = Xs @ Wtot.T

        loss = loss_fn(y_pred, ys)
        losses.append(loss.item())

        if not t % args.print_freq:
            print(t, loss.item())
            
        yt_pred = Xt @ Wtot.T

        risk = risk_fn(yt_pred, yt)
        risks.append(risk.item())
        
        if args.low_rank_eval:
            losses_low.append(np.array([loss_fn(Xs_l @ Wtot.T, ys).item() for Xs_l in Xs_low]))
        if args.ind_eval:
            losses_ind.append(np.array([loss_fn((Xs[i, :].reshape(1, -1) @ Wtot.T).squeeze(), ys[i]).item() for i in range(args.samples)]))
        if args.weight_eval:
            assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
            weight_mse.append((Wtot.squeeze()-ws.squeeze())**2)
        if args.weight_eval_min:
            assert args.linear and args.no_bias, "Weight evaluation not appropriate for non-linear model or model with bias"
            weight_mse_min.append((Wtot.squeeze()-w_min.squeeze())**2)
        if args.save_weights:
            weights.append(np.column_stack((u_track[-1], z_track[-1])))

        if not t % args.print_freq:
            print(t, risk.item())
            
            
    return {"loss": np.array(losses), "risk": np.array(risks), "weight_norm": weights_norm,
            "eigenvals": np.array([]), "grad_norm": grad_norms, "losslowrank": np.row_stack(losses_low) if losses_low else np.array(losses_low),
            "losses_ind": np.row_stack(losses_ind) if losses_low else np.array(losses_ind),
            "weight_mse": np.row_stack(weight_mse) if weight_mse else np.array(weight_mse), 
            "weight_mse_min": np.row_stack(weight_mse_min) if weight_mse_min else np.array(weight_mse_min),
            "weights": np.row_stack(weights) if weights else np.array(weights)}



# With actual input data
def dt(u, z, S, St):
    assert S.shape == z.shape
    return (St - u * z * S)


def dzdt(u, z, S, St):
    return 2 * u * dt(u, z, S, St)


def dudt(u, z, S, St):
    return 2 * (dt(u, z, S, St) @ z.T).squeeze()


# Sampling only noise in output (and assuming that we know the true weights)
def dt_s(u, z, S, beta, eps):
    assert S.shape == z.shape
    return (beta - u * z) * S + eps * S**0.5


def dzdt_s(u, z, S, beta, eps):
    return 2 * u * dt_s(u, z, S, beta, eps)


def dudt_s(u, z, S, beta, eps):
    return 2 * (dt_s(u, z, S, beta, eps) @ z.T).squeeze()

