from core.controller import SystemController
import numpy as np

from utils.sampling import SobolGenerator

def main():
    problem_name="SIMPLEPOLY"
    controller = SystemController(problem_name)
    
    config = controller.get_mission_config()
    sobol_gen = SobolGenerator(config)
    initial_batch = sobol_gen.generate(n_samples=100)   
    print("Initial Batch of Points (First 5):")
    print([tuple(round(float(v), 3) for v in row) for row in initial_batch.xs[:]])
    print(f"Mission Loaded: {problem_name}")
    print(f"Dimensions: {config.n_dim}")
    print(config)
    print(f"Bounds (First 2 dims): Lower={config.lower_bound[:2]}, Upper={config.upper_bound[:2]}")

if __name__ == "__main__":
    main()