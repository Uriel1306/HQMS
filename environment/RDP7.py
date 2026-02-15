import numpy as np
from schemas.data_types import GlobalConfig, PointBatch, EvaluationInput, EvaluationOutput

class RJY3_Problem:
    def __init__(self, d_dim=10, m_obj=2):
        self.D = d_dim
        self.M = m_obj
        
        self.taut = 10
        self.nt = 10
        
        rng = np.random.default_rng(seed=42)
        self.lower_limit = 0
        self.upper_limit = 2 * self.nt + 1
        
        perm = rng.permutation(self.upper_limit) 
        self.random_integers = perm + self.lower_limit
        
        self.FE = 0
        self.N = 100

    def get_config(self) -> GlobalConfig:

        lower = np.array([0.0] + [-1.0] * (self.D - 1))
        upper = np.array([1.0] + [1.0] * (self.D - 1))
        
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