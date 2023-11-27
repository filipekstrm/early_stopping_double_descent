import numpy as np


def test_transform_data_w_kappa():

    # Sample some data
    d = np.random.randint(low=2, high=100)
    n = np.random.randint(low=(d+1), high=1000)  # Assume n > d
    
    sigmas = np.random.uniform(low=0.01, high=10.0, size=(d,))
    X = np.random.randn(n * d).reshape(n, d) * sigmas.reshape(1, -1)

    # SVD
    U, S, Vh = np.linalg.svd(X, full_matrices=True)

    # Transform data
    kappas = np.geomspace(0.01, 100.0, num=1000)
    kappa = np.random.choice(kappas)  #np.random.uniform(low=0.01, high=100.0)

    S_inv = np.diag(1 / S)
        
    p = int(np.ceil(d/2))  # Number of dimensions with eigenvalue equal to 1
    F = np.diag(np.sort(np.concatenate((np.ones((p,)), (np.ones((d-p,))) * kappa)))[::-1])  # Stretch/squeeze some dims
    
    transform_mat = np.transpose(Vh)@S_inv@F
    
    Z = X @ transform_mat
    
    # New SVD
    Uz, Sz, Vzh = np.linalg.svd(Z, full_matrices=True)

    # Check eigenvalues
    eps = 1e-5
    eig_bool_kappa = (np.abs(Sz[p:] - kappa).sum() < eps) if kappa < 1 else (np.abs(Sz[:(d-p)] - kappa).sum() < eps)
    assert eig_bool_kappa
    
    eig_bool_other = (np.abs(Sz[:p] - 1.0).sum() < eps) if kappa < 1 else (np.abs(Sz[(d-p):] - 1.0).sum() < eps)
    print(np.abs(Sz[(p-1):] - 1.0).sum())
    assert eig_bool_other
    
    # Check singular vectors
    Z_recon = U[:, :d] @ np.diag(Sz)
    
    assert np.abs(Z - Z_recon).sum() < eps
    

if __name__ == '__main__':
    test_transform_data_w_kappa()