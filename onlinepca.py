import numpy as np
from scipy.spatial import distance

class OnlineWeightedPCA:

    def __init__(self, dimension:int, n_components:int=2, mean=None, cov=None, weight:float=None, explained_variance = None, components = None, proc_cov:bool = True):

        self._dimension = dimension

        weight = 1 if weight is None else weight

        if mean is None:
            self._mean = np.zeros(self._dimension, dtype="float64")
        else:
            self._mean = mean.astype(np.float64)

        if cov is None:
            self._cov = np.zeros((self._dimension, self._dimension), dtype="float64")
        else:
            self._cov = cov.astype(np.float64)

        self._t = 0 if weight is None else 1
        self._n = weight

        if explained_variance is None:
            self.explained_variance_ = np.zeros(n_components, dtype="float64")
            self.explained_variance_ratio_ = np.zeros(n_components, dtype="float64")
        else:
            self.explained_variance_ = explained_variance[:n_components].astype(np.float64)
            self.explained_variance_ratio_ = (self.explained_variance_ / np.sum(self.explained_variance_)).astype(np.float64)
            
        if components is None:
            self.components_ = np.array([vector/vector.sum() for vector in np.ones((n_components, self._dimension))], dtype="float64")
        else:
            self.components_ = components[:n_components].astype(np.float64)
        
        self.proc_cov = proc_cov
    
    def set_sklearn(self, model, w = 1, t = 1):
        
        self.components_ = model.components_
        self.explained_variance_ = model.explained_variance_
        self.explained_variance_ratio_ = model.explained_variance_ratio_
        self._mean = model.mean_
        self.n_components_ = model.n_components_
        self._cov = model.get_covariance()
        self._n = w
        self._t = t

    def add(self, f:list, p:float = 1):

        f = np.array(f)
        q:float = 1 - p
        mean = p * f + q * self._mean
        
        if self._t > 0:

            mean_ = self._mean
            f_new = f-mean
            f_old = f-mean_
            
            if not self.proc_cov: 
                for i in range(self._dimension): 
                    self._cov[i, i] = p * f_new[i] * f_old[i] + q * self._cov[i, i]
            else: 
                for j in range(self._dimension):
                    for i in range(j, self._dimension):
                        self._cov[j, i] = p * f_new[i] * f_old[j] + q * self._cov[j, i]
                        self._cov[i, j] = self._cov[j, i]

        self._mean = mean

        self._t += 1
        
        return self
        
    def __iadd__(self, f): 
        return self.average(f)

    def n(self, ): return self._n
    def __len__(self): return self._t

    def mean(self,): return self._mean
    def var(self, ): return np.diag(self._cov)
    def cov(self, ): return self._cov
    def std(self, ): return np.sqrt(self.var())

    def average(self, f, w=1):
        p = w / (self._n + w)
        tmp = self.add(f, p)
        self._n = w + self._n
        return tmp
        
    def ema(self, f, alpha:float=0.5):
        tmp = self.add(f, alpha)
        w = self._n * alpha / (1-alpha)
        self._n = w + self._n
        return tmp

    def fit(self, tol = 1e-10, max_iter = 1000):
        
        value_list = []
        vector_list = []
        matrix = self.cov()

        for eigen_value, eigen_vector in zip(self.explained_variance_, self.components_):
            eigen_value_ = eigen_value.copy()
            eigen_vector_ = eigen_vector.copy()
            for _ in range(max_iter):
                vector = matrix @ eigen_vector_
                value = (vector @ vector) / (vector @ eigen_vector_)
                if np.abs(eigen_value_ - value) < tol: break
                eigen_vector_ = vector / np.linalg.norm(vector)
                eigen_value_ = value
                

            value_list.append(eigen_value_)
            vector_list.append(eigen_vector_)

            matrix = matrix - eigen_value_ * np.transpose([eigen_vector_]) @ [eigen_vector_]

        index = np.argsort(-np.abs(value_list))
        self.explained_variance_ = np.array(value_list, dtype="float64")[index]
        self.components_ = np.array(vector_list, dtype="float64")[index]
        
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        
    def transform(self, X):

        mean = self.mean()
        white_explained_variance_ = np.diag(1./np.sqrt(self.explained_variance_))
        return np.array([white_explained_variance_ @ np.dot(self.components_, (x - mean)) for x in X])
    
    
    def __call__(self, x): return np.diag(1./np.sqrt(self.explained_variance_)) @ np.dot(self.components_, (x - self.mean()))

    def T2(self, x):
        x_ = x - self.mean()
        return distance.mahalanobis(f, self._mean, np.linalg.pinv(self._cov))**2

    def anomaly(self, x):
        x_ = x - self.mean()
        return x_.T @ x_ - x_.T @ self.components_.T @ self.components_ @ x_