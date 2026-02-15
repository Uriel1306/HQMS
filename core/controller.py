import numpy as np
from typing import Callable, Optional
from factory import ProblemFactory
from schemas.data_types import PointBatch
from core.referee import CompetitionRefree


class SystemController:
    def __init__(self, problem_name: str):
        self.problem = ProblemFactory.create_problem(problem_name)
        self.config = self.problem.get_config()
        self.referee = CompetitionRefree()
        self.pop_getter: Optional[Callable[[], PointBatch]] = None

    def set_population_getter(self, func: Callable[[], PointBatch]):
        self.pop_getter = func

    def evaluate_batch(self, batch: PointBatch) -> PointBatch:
        xs = batch.xs
        total = xs.shape[0]
        idx = 0
        results = {'fs': [], 'cvs': [], 'gs': []}
        N = self.problem.N

        while idx < total:
            curr_fe = self.problem.FE
            next_check = (int(curr_fe / N) + 1) * N
            dist = next_check - curr_fe if next_check > curr_fe else N

            take = min(total - idx, dist)

            chunk = self.problem.evaluate(PointBatch(xs=xs[idx: idx + take]))

            results['fs'].append(chunk.fs)
            if chunk.cvs is not None:
                results['cvs'].append(chunk.cvs)
            if chunk.gs is not None:
                results['gs'].append(chunk.gs)

            idx += take

            if self.problem.FE % N == 0:
                self._measure()

        return PointBatch(
            xs=xs,
            fs=np.vstack(results['fs']),
            cvs=np.vstack(results['cvs']) if results['cvs'] else None,
            gs=np.vstack(results['gs']) if results['gs'] else None
        )

    def _measure(self):
        if self.pop_getter:
            self.referee.update_metrics(
                self.pop_getter(),
                self.problem.get_current_optimum(1000)
            )

    def get_final_results(self):
        return self.referee.get_final_metrics()

    @property
    def current_fe(self) -> int:
        return self.problem.FE

    @property
    def max_fe(self) -> int:
        return self.problem.N * (30 * self.problem.taut + 50)
    