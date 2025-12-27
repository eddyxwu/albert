```python
import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Lagrangian:
    L = R - (1/4) * F_μν F^μν + α * R * F_μν F^μν

    This theory generalizes the Einstein-Maxwell action by introducing a non-minimal 
    coupling between the Ricci scalar R and the Maxwell invariant F^2 = F_μν F^μν.
    The metric components are derived as a generalized Kerr-Newman solution 
    where the non-minimal coupling alpha modifies the effective charge contribution 
    to the spacetime curvature. In the limit alpha -> 0, the theory recovers 
    the standard Kerr-Newman solution.
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def get_metric(self, r, M, a, Q):
