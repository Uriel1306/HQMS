import numpy as np
from schemas.data_types import GlobalConfig, PointBatch

class SimplePoly_Problem:
    def __init__(self, d_dim=2, m_obj=1):
        self.D = 2 

        self.M = 1 
        
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
        lower = np.full(self.D, -1.0)
        upper = np.full(self.D, 1.0)
        
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
        
        x = X[:, 0]
        y = X[:, 1]
        

        f1 = x**2 + y**2 + 2*x*y - y - x
        

        fs = f1.reshape(-1, 1)
        
        self.FE += n_points
        
        g_debug = f1 + 0.25
        
        return PointBatch(xs=X, fs=fs, cvs=None, gs=g_debug.reshape(-1, 1))

    def get_current_optimum(self, n_points=100) -> PointBatch:
        x_vals = np.linspace(-0.5, 1.0, n_points)
        y_vals = 0.5 - x_vals
        
        xs = np.column_stack((x_vals, y_vals))
        
        f1 = np.full(n_points, -0.25)
        fs = f1.reshape(-1, 1)
        
        return PointBatch(xs=xs, fs=fs)