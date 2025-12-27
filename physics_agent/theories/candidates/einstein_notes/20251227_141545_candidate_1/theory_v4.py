```python
import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Lagrangian:
    L = R - (1/4) * F_μν F^μν + α * R * F_μν F^μν

    This theory extends the Einstein-Hilbert action with a Maxwell term and a 
    non-minimal coupling between the Ricci scalar R and the Maxwell invariant F^2. 
    The term α * R * F^2 introduces a coupling between the curvature and the 
    electromagnetic field, leading to corrections in the metric that scale with 
    higher powers of 1/r. In the weak field limit, this modifies the 