import numpy as np
from schemas.data_types import GlobalConfig, PointBatch

class RDP7_Problem:
    def __init__(self, d_dim=10, m_obj=2):
        self.D = d_dim
        self.M = m_obj
        
        self.taut = 10
        self.nt = 10
        
        self.FE = 0
        self.N = 100
        
        self.lower_limit = 0
        self.upper_limit = 2 * self.nt + 1
        
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(self.upper_limit)
        self.random_integers = perm + self.lower_limit
        
    def get_config(self) -> GlobalConfig:
        lower = np.zeros(self.D)
        lower[0] = 0.0
        lower[1:] = -1.0
        
        upper = np.ones(self.D)
        
        return GlobalConfig(
            n_dim=self.D,
            n_objectives=self.M,
            n_constraints=0,
            lower_bound=lower,
            upper_bound=upper
        )

    def _calculate_t(self):
        index_t = int(np.floor(self.FE / self.N / self.taut)) % self.upper_limit
        Q_t = self.random_integers[index_t]
        t = Q_t / self.nt
        return t

    def evaluate(self, batch: PointBatch) -> PointBatch:
        X = batch.xs
        n_points = X.shape[0]
        
        t = self._calculate_t()
        
        A = 0.05
        W = np.floor(6 * np.sin(0.5 * np.pi * (t - 1)))
        a_param = np.floor(100 * (np.sin(0.5 * np.pi * t))**2)
        
        y = X.copy()
        y[:, 0] = np.abs(X[:, 0] * np.sin((2 * a_param + 0.5) * np.pi * X[:, 0]))
        
        term = (X[:, 1:]**2 - X[:, :-1])**2
        g = np.sum(term, axis=1)
        
        f1 = (1 + g) * (y[:, 0] + A * np.sin(W * np.pi * y[:, 0]))
        f2 = (1 + g) * (1 - y[:, 0] + A * np.sin(W * np.pi * y[:, 0]))
        
        fs = np.column_stack((f1, f2))
        
        self.FE += n_points
        
        return PointBatch(xs=X, fs=fs, cvs=None, gs=g.reshape(-1, 1))

    def get_current_optimum(self, n_points=1000) -> PointBatch:
        t = self._calculate_t()
        W = np.floor(6 * np.sin(0.5 * np.pi * (t - 1)))
        A = 0.05
        
        x = np.linspace(0, 1, n_points)
        
        f1 = x + A * np.sin(W * np.pi * x)
        f2 = 1 - x + A * np.sin(W * np.pi * x)
        
        fs = np.column_stack((f1, f2))
        xs = np.zeros((n_points, self.D))
        xs[:, 0] = x 
        
        return PointBatch(xs=xs, fs=fs)