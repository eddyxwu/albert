import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    category = "quantum"
    
    def __init__(self, coupling=0.1):
        super().__init__(name="Quantum Unified Theory")
        self.coupling = coupling
        
        # Define Lagrangian symbols
        R = sp.Symbol('R')  # Ricci scalar
        F = sp.Symbol('F')  # Maxwell invariant F_mu_nu F^mu_nu
        
        # Lagrangian: Einstein-Hilbert + Maxwell + Quantum Correction (R-F coupling)
        # This represents a non-minimal coupling often found in effective field theories
        self.lagrangian = R - 0.25 * F + self.coupling * R * F
        
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        alpha = self.coupling
        
        # Metric modified by electromagnetic and quantum correction terms.
        # The term proportional to 1/r^2 mimics a Reissner-Nordstrom (EM) effect,
        # while the 1/r^4 term introduces a quantum-corrected vacuum polarization behavior.
        f_r = 1 - rs/r + (alpha * rs**2) / (r**2) - (alpha**2 * rs**3) / (r**4)
        
        # Ensure metric components are calculated via torch tensors
        g_tt = -f_r
        g_rr = 1.0 / f_r
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp