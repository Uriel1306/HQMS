import numpy as np
from schemas.data_types import GlobalConfig, PointBatch

class RDCP1_Problem:
    def __init__(self, d_dim=10, m_obj=2):
        self.D = d_dim
        self.M = m_obj
        self.N = 100
        self.taut = 10
        self.nt = 10
        self.FE = 0
        
        self.burn_in = 500
        total_steps = 30
        self.expected_max_fe = (total_steps * self.taut * self.N) + self.burn_in
        
        self.upper_limit = total_steps + 1
        self.lower_limit = 0
        
        rng = np.random.default_rng(seed=42)
        self.random_integers = rng.permutation(self.upper_limit) + self.lower_limit + 1

    def get_config(self) -> GlobalConfig:
        return GlobalConfig(
            n_dim=self.D,
            n_objectives=self.M,
            n_constraints=1,
            lower_bound=np.zeros(self.D),
            upper_bound=np.ones(self.D)
        )

    def _get_t(self):
        steps_passed = int(np.floor(self.FE / (self.N * self.taut)))
        idx = steps_passed % self.upper_limit
        return self.random_integers[idx] / self.nt

    def _calc_cv(self, f1, f2):
        theta = -0.15 * np.pi
        part1 = np.sin(theta) * f2 + np.cos(theta) * f1
        part2 = -np.cos(theta) * f2 + np.sin(theta) * f1
        return (2 * np.sin(5 * np.pi * part1))**6 + part2

    def evaluate(self, batch: PointBatch) -> PointBatch:
        X = batch.xs
        self.FE += X.shape[0]
        
        t = self._get_t()
        G = np.abs(np.sin(0.5 * np.pi * t))
        g = 1 + np.sum((X[:, 1:] - G)**2, axis=1)
        
        f1 = g * X[:, 0] + G
        f2 = g * (1 - X[:, 0]) + G
        
        f1[f1 < 1e-18] = 0
        f2[f2 < 1e-18] = 0
        
        cv = self._calc_cv(f1, f2)
        
        return PointBatch(
            xs=X, 
            fs=np.column_stack((f1, f2)), 
            cvs=cv.reshape(-1, 1), 
            gs=g.reshape(-1, 1)
        )

    def get_current_optimum(self) -> PointBatch:
        t = self._get_t()
        G = np.abs(np.sin(0.5 * np.pi * t))
        
        x = np.linspace(0, 1, 500)
        
        addition1 = np.arange(0.001, 0.401, 0.001)
        
        P1_main = x + G
        P2_main = 1 - x + G
        
        min_P1 = np.min(P1_main)
        max_P1 = np.max(P1_main)
        
        addition2 = np.full(len(addition1), min_P1)
        addition3 = addition1 + max_P1
        
        P1 = np.concatenate([P1_main, addition2, addition3])
        P2 = np.concatenate([P2_main, addition3, addition2])
        
        c_vals = self._calc_cv(P1, P2)
        mask = c_vals <= 0
        
        valid_f1 = P1[mask]
        valid_f2 = P2[mask]
        
        xs = np.zeros((len(valid_f1), self.D))
        xs[:, 0] = 0 
        xs[:, 1:] = G 
        
        return PointBatch(xs=xs, fs=np.column_stack((valid_f1, valid_f2)))