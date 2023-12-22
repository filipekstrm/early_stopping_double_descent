import numpy as np
from scipy.stats import ortho_group



class linear_model():
    def __init__(self, d, sigma_noise=0, beta=None, scale_beta=False, normalized=True, sigmas=None, s_range=[1,10], coupled_noise=False, transform_data=False, kappa=None, p=None, cont_eigs=False, zero_eigs=0):
        self.d = d
        if beta is None:
            self.beta = np.random.randn(self.d)
            #self.beta = np.ones(self.d)
        else:
            self.beta = beta
            assert beta.shape[0] == self.d
            
      
        self.sigma_noise = sigma_noise
        self.coupled_noise = coupled_noise
        self.transform_data = transform_data
        self.transform_mat = None
        self.right_singular_vecs = None
        self.kappa = kappa
        self.cont_eigs = cont_eigs
        self.zero_eigs = zero_eigs
        self.p = p if p is not None else default_p(d, zero_eigs) # The number of eigenvalues equal to 1

        if kappa is not None and scale_beta:
            F = self.modulation_matrix()
            
            if self.zero_eigs > 0:
                self.beta = np.concatenate((np.linalg.inv(F)@self.beta[:-self.zero_eigs], self.beta[-self.zero_eigs:])) 
            else:
                self.beta = np.linalg.inv(F)@self.beta
       
        if is_float(sigmas):
            sigmas = float(sigmas)
            if normalized:
                self.sigmas = np.array([sigmas] * d) / np.sqrt(self.d)
            else:
                self.sigmas = np.array([sigmas] * d) 
                
        elif isinstance(sigmas, list):
            
            assert len(sigmas) == d
            self.sigmas = np.array(sigmas)
        
        elif sigmas in ['geo', 'geometric']:
            if normalized:
                self.sigmas = np.geomspace(s_range[0], s_range[1], d) / np.sqrt(self.d)
            else:
                self.sigmas = np.geomspace(s_range[0], s_range[1], d)            
        
        elif sigmas is None:
            if normalized:
                self.sigmas = (np.array([1 for i in range(int(np.floor(d/2)))] +
                                    [0.01 for i in range(int(np.ceil(d/2)))]) / np.sqrt(self.d))
            else:
                self.sigmas = np.array([1 for i in range(int(np.floor(d/2)))] +
                                    [0.01 for i in range(int(np.ceil(d/2)))])
        else:
            self.sigmas = sigmas
            
    
    def sample(self, n, train=True): 
    
        Xs = np.random.randn(n * self.d).reshape(n, self.d) * self.sigmas.reshape(1, -1)
        
        if self.transform_data or self.kappa:
            if train:
                U, S, Vh = np.linalg.svd(Xs, full_matrices=True)
                self.right_singular_vecs = U
            
                if self.transform_data:
                    self.transform_mat = np.transpose(Vh) 
                   
                if self.kappa:
                    
                    # TODO: do this in a nicer way
                    if n >= self.d:
                        S_inv = np.diag(1 / S)
                        F = self.modulation_matrix() #np.diag(np.sort(np.concatenate((np.ones((self.p,)), (np.ones((self.d-self.p,))) * self.kappa)))[::-1])  # Stretch/squeeze some dims

                    else:
                    #    print("Can not yet handle the case d > n")
                        S_inv = np.diag(np.concatenate((1 / S, np.zeros((self.d - n,))), axis=-1))
                        F = np.diag(np.concatenate((np.diag(self.modulation_matrix()), np.zeros((self.d - n,))), axis=-1))
                   
           
                    self.transform_mat = np.transpose(Vh)@S_inv@F@Vh if self.transform_mat is None else self.transform_mat@S_inv@F 
                    
            else:
                assert self.transform_mat is not None, "You need to sample training data first for transform to be possible"
                
            Xs = Xs @ self.transform_mat
            _, S, Vh = np.linalg.svd(Xs, full_matrices=True)
            print(S)
            print(Vh)

        # TODO: INSER ATT I "WHEN AND HOW..." SÅ MÅSTE DE ANTA X \in R^{DxN} (OCH HÄR HAR VI X \in R^{NxD}). MEN DETTA ÄR OCKSÅ FÖRVIRRANDE; FÖR VILKET RUM TRANSFORMERAR VI DÅ DATAN TILL?
        if self.coupled_noise:
        
            if isinstance(self.sigma_noise, float):
                sigma_noise = [self.sigma_noise, 0.0]
            else:
                assert len(self.sigma_noise) == 2, "Noise in output should be float, or list of length 2"
                sigma_noise = self.sigma_noise
                    
            ys = Xs @ self.beta 
            
            if train: # TODO: because I assume that the test set is noise less 
                U, S, Vh = np.linalg.svd(Xs, full_matrices=True)

                #assert np.abs((Xs - (U @ diag(S) @ Vh))).sum() < 1e-5
                
                z = np.zeros((S.shape[0],))
                z[S**2 >= 1] = sigma_noise[0]*np.random.randn((S**2 >= 1).sum())
                z[S**2 < 1] = sigma_noise[1]*np.random.randn((S**2 < 1).sum())
                
                #if self.transform_data:
                #    V = self.transform_mat
                #else:
                V = np.transpose(Vh)
                    
                if self.d < n:
                    assert np.isclose(sigma_noise[1], 0.0, rtol=1e-10), "Case d < n does not yet handle two noise levels"

                    eps = np.concatenate((V @ z, np.zeros((n - S.shape[0],)))) # TODO: it does not matter if we extend V, as the eigenvalues are still 0 (and hence z is 0) for those columns?
                elif self.d == n:
                    eps = V @ z
                else:
                    eps = (V @ np.concatenate((z, sigma_noise[1]*np.random.randn(self.d - n))))[:n] # TODO: this is probably not how to do it? Also when considering noise in all dirs?
                
                
                ys += eps 
                
        else:

            if isinstance(self.sigma_noise, float):
                sigma_noise = [self.sigma_noise, self.sigma_noise]
            else:
                sigma_noise = self.sigma_noise
            
            ys = []
            for i in range(n):
                y = Xs[i, :] @ self.beta #+ sigma_noise*np.random.randn(1)[0] # TODO: noise free test data?
                ys += [y]
                
            ys = np.array(ys)
            
            # NOTE: I HAVE MADE CHANGES HERE (TEST DATA WAS NOT NOISE FREE BEFORE)
            # WOULD IT MAKE SENSE TO SET THE LOWER NOISE LEVEL FOR TEST?
            if train: 
                split = int(np.ceil(n / 2))
                eps = np.concatenate([sigma_noise[0]*np.random.randn(split), sigma_noise[1]*np.random.randn(n - split)])
                np.random.shuffle(eps)
                ys += eps
        
        return Xs, ys
        
    def modulation_matrix(self):
    
        if self.zero_eigs > 0:
            return np.diag(get_modulation_matrix(self.d - self.zero_eigs, self.p, self.kappa, diag_sorted=True, cont_eigs=self.cont_eigs)[::-1])
        else:
            return np.diag(get_modulation_matrix(self.d, self.p, self.kappa, diag_sorted=True, cont_eigs=self.cont_eigs)[::-1])

 
def get_modulation_matrix(d, p, k, diag_sorted=False, cont_eigs=False):
    
    
    if cont_eigs:
        S = np.diag(np.linspace(1.0, k, num=d))
    else:
        S = np.eye(d)
        S[:p, :p] *= 1
        S[p:, p:] *= k
    
    if diag_sorted:  # Make diagonal matrix with sorted diagonal
        F = np.sort(np.diag(S)) 
    else:
        U = ortho_group.rvs(d)
        VT = ortho_group.rvs(d)
        F = np.dot(U, np.dot(S, VT))
    
    return F
    
    
    
def default_p(dim, zero_eigs):
    return int(np.ceil((dim - zero_eigs)/2))
    
    
 
def is_float(element: any) -> bool:

    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
        