from pydantic import BaseModel, Field, ConfigDict, model_validator
import numpy as np
from typing import List, Optional, Tuple

class ArbitraryModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class GlobalConfig(ArbitraryModel):
    n_dim: int = 10
    n_objectives: int = 2
    n_constraints: int = 0
    
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None

    @model_validator(mode='after')
    def set_default_bounds(self):
        if self.lower_bound is None:
            self.lower_bound = np.zeros(self.n_dim)
        if self.upper_bound is None:
            self.upper_bound = np.ones(self.n_dim)
            
        if len(self.lower_bound) != self.n_dim:
            raise ValueError(f"Lower bound length {len(self.lower_bound)} != n_dim {self.n_dim}")
        if len(self.upper_bound) != self.n_dim:
            raise ValueError(f"Upper bound length {len(self.upper_bound)} != n_dim {self.n_dim}")
            
        return self
    @property
    def samples_per_step(self) -> int:
        return self.n_dim + 1

class TankConfig(ArbitraryModel):
    tank_id: int
    objective_weights: np.ndarray
    initial_radius: float = 0.2
    total_steps: int = 15
    radius_decay_rate: float = 0.5 

class Point(ArbitraryModel):
    x: np.ndarray
    f: Optional[np.ndarray] = None
    cv: Optional[np.ndarray] = None
    
    debug_g: Optional[float] = None

    @property
    def is_feasible(self) -> bool:
        if self.cv is None: return True
        violation = np.sum(np.maximum(0, self.cv))
        return violation < 1e-6

    def __repr__(self):
        return f"Point(X={np.round(self.x, 2)}, Feas={self.is_feasible})"

class PointBatch(ArbitraryModel):
    xs: np.ndarray
    fs: Optional[np.ndarray] = None
    cvs: Optional[np.ndarray] = None
    gs: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self.xs)

    def filter_feasible(self) -> 'PointBatch':
        if self.cvs is None:
            return self

        violations = np.sum(np.maximum(0, self.cvs), axis=1)
        mask = violations < 1e-6
        new_xs = self.xs[mask]
        new_fs = self.fs[mask] if self.fs is not None else None
        new_cvs = self.cvs[mask]
        new_gs = self.gs[mask] if self.gs is not None else None
        
        return PointBatch(xs=new_xs, fs=new_fs, cvs=new_cvs, gs=new_gs)

    def calculate_weighted_scores(self, weights: np.ndarray) -> np.ndarray:
        if self.fs is None:
            return np.full(self.count, np.inf)
            
        scores = np.dot(self.fs, weights)
        
        if self.cvs is not None:
            violations = np.sum(np.maximum(0, self.cvs), axis=1)
            penalty_mask = violations > 1e-6
            scores[penalty_mask] += 1e9 + violations[penalty_mask] * 1000
            
        return scores

    def to_point_list(self) -> List[Point]:
        points = []
        for i in range(self.count):
            p = Point(
                x=self.xs[i],
                f=self.fs[i] if self.fs is not None else None,
                cv=self.cvs[i] if self.cvs is not None else None,
                debug_g=self.gs[i] if self.gs is not None else None
            )
            points.append(p)
        return points

class TankState(ArbitraryModel):
    config: TankConfig
    current_point: Point
    current_radius: float
    
    history: List[Point] = []
    steps_taken: int = 0

    @property
    def steps_remaining(self) -> int:
        return self.config.total_steps - self.steps_taken

    def add_to_history(self, point: Point):
        self.history.append(point)

    def advance_step(self):
        self.steps_taken += 1