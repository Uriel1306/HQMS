import numpy as np
from scipy.stats import qmc
from schemas.data_types import GlobalConfig, PointBatch

class SobolGenerator:

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.d = config.n_dim
        

        self.sampler = qmc.Sobol(d=self.d, scramble=True, seed=42)

    def generate(self, n_samples: int) -> PointBatch:

        samples_unit = self.sampler.random(n_samples)
        

        lower = self.config.lower_bound
        upper = self.config.upper_bound
        

        scaled_samples = qmc.scale(samples_unit, lower, upper)
        
        return PointBatch(xs=scaled_samples)

    def reset(self):
        self.sampler.reset()
