# HQMS: Hierarchical Quasi-random Multi-directional Search

## Overview
HQMS is a reactive, memory-less optimization algorithm designed for **Dynamic Multi-Objective Optimization Problems (DMOPs)** under strict computational budgets. This implementation was developed for the **WCCI 2026 Competition**.

The algorithm addresses high-dimensional, constrained landscapes by combining global quasi-random sampling with localized, multi-scale gradient estimation.

## Key Features
- **Hierarchical Sampling**: Initial global mapping using Sobol Sequences to ensure spatial coverage in 10D space.
- **Multi-directional "Hedgehog" Search**: Five localized agents (Tanks) utilizing non-symmetrical radial sampling (Zigzag) to detect both feasibility boundaries and local gradients.
- **Constraint Extrapolation**: A ballistic repair mechanism (Wall-Jump) for escaping infeasible regions without redundant function evaluations.
- **Reactive Strategy**: Full state-reset upon environmental change detection to prevent negative knowledge transfer in chaotic dynamics.

## Technical Requirements
- Python 3.9+
- NumPy
- SciPy (for `scipy.stats.qmc.Sobol`)
- (Optional) MATLAB Engine for Python (for benchmark validation)

## Installation
```bash
pip install -r requirements.txt
