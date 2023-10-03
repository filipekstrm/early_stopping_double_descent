import numpy as np



class linear_model():
    def __init__(self,d,sigma_noise=0,beta=None,sigmas=None,normalized=True,s_range=[1,10], coupled_noise=False, transform_data=False):
        self.d = d
        if beta is None:
            self.beta = np.random.randn(self.d)
            #self.beta = np.ones(self.d)
        else:
            self.beta = beta
        

        self.sigma_noise = sigma_noise
        self.coupled_noise = coupled_noise
        self.transform_data = transform_data
        self.transform_mat = None
        
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
            
    def estimate_risk(self,estimator,avover=500):
        # estimator is an instance of a class with a predict function mapping x to a predicted y
        # function estimates the risk by averaging
        risk = 0
        for i in range(avover):
            x = np.random.randn(self.d) * self.sigmas 
            y = x @ self.beta + self.sigma_noise*np.random.randn(1)[0]
            risk += (y - estimator.predict(x))**2
        return risk/avover
    
    def compute_risk(self,hatbeta):
        # compute risk of a linear estimator based on formula
        return np.linalg.norm( self.beta - hatbeta )**2 + self.sigma_noise**2
    
    def sample(self, n, train=True): 
    
        Xs = np.random.randn(n * self.d).reshape(n, self.d) * self.sigmas.reshape(1, -1)
        
        if self.transform_data:
            if train:
                U, S, Vh = np.linalg.svd(Xs, full_matrices=True)
                self.transform_mat = np.transpose(Vh)
            else:
                assert self.transform_mat is not None, "You need to sample training data first for transform to be possible"
                
            Xs = Xs @ self.transform_mat

        if self.coupled_noise:
        
            if not isinstance(self.sigma_noise, float):
                sigma_noise = [self.sigma_noise, 0.0]
            else:
                assert len(self.sigma_noise) == 2, "Noise in output should be float, or list of length 2"
                sigma_noise = self.sigma_noise
        
            ys = Xs @ self.beta 
            
            if train: # TODO: because I assume that the test set is noise less 
                U, S, Vh = np.linalg.svd(Xs, full_matrices=True)

                #assert np.abs((Xs - (U @ diag(S) @ Vh))).sum() < 1e-5
                
                z = np.zeros((S.shape[0],))
                z[S**2 >= 1] = self.sigma_noise[0]*np.random.randn((S**2 >= 1).sum())
                z[S**2 < 1] = self.sigma_noise[1]*np.random.randn((S**2 < 1).sum())
                
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
                    eps = (V @ np.concatenate((z, self.sigma_noise[1]*np.random.randn(self.d - n))))[:n] # TODO: this is probably not how to do it? Also when considering noise in all dirs?
                
                
                ys += eps 
                
        else:
            ys = []
            
            if not isinstance(self.sigma_noise, float):
                sigma_noise = self.sigma_noise[0]
            else:
                sigma_noise = self.sigma_noise
        
            for i in range(n):
                y = Xs[i, :] @ self.beta + sigma_noise*np.random.randn(1)[0] # TODO: noise free test data?
                ys += [y]
                
            ys = np.array(ys)
        
        return Xs, ys
        
        
def is_float(element: any) -> bool:

    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False