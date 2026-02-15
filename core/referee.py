import numpy as np
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from schemas.data_types import PointBatch

class CompetitionRefree:
    def __init__(self):
        self.history_igd = []
        self.history_hv = []

    def update_metrics(self, obtained_population: PointBatch, true_optimum: PointBatch):
        true_pf = true_optimum.fs
        obtained_pf = obtained_population.fs
        
        if true_pf is None or obtained_pf is None:
            return

        try:
            igd_indicator = IGD(true_pf)
            self.history_igd.append(float(igd_indicator(obtained_pf)))
        except Exception:
            pass
        
        try:
            num_obj = true_pf.shape[1]
            ref_point = np.array([1.1] * num_obj)
            hv_indicator = HV(ref_point=ref_point)
            self.history_hv.append(float(hv_indicator(obtained_pf)))
        except Exception:
            pass

    def get_final_metrics(self):
        igd_mean = np.mean(self.history_igd) if self.history_igd else 0.0
        igd_std = np.std(self.history_igd) if self.history_igd else 0.0
        hv_mean = np.mean(self.history_hv) if self.history_hv else 0.0
        hv_std = np.std(self.history_hv) if self.history_hv else 0.0        
        return {
            "MIGD": float(igd_mean),
            "MHV": float(hv_mean),
            "STD_IGD": float(igd_std),
            "STD_HV": float(hv_std)
        }