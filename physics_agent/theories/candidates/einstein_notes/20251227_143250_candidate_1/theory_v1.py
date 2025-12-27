import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class QuantumUnifiedTheory(GravitationalTheory):
    category = "quantum"
    
    def __init__(self, coupling=0.1, quantum_scale=1e-35):
        super().__init__(name="Quantum Unified Theory")
        self.coupling = coupling
        self.ell = quantum_scale # Planck-scale regularization parameter
        
        # Define Lagrangian symbols
        R = sp.Symbol('R')  # Ricci scalar
        F_sq = sp.Symbol('F_sq')  # Maxwell invariant (F_munu F^munu)
        phi = sp.Symbol('phi')  # Scalar field
        
        # Lagrangian unifying gravity, electromagnetism and a quantum correction term
        # Includes a non-minimal coupling between the Ricci scalar and the gauge field
        self.lagrangian = R - 0.25 * F_sq + self.coupling * (R * F_sq)
        
    def get_metric(self, r, M_param, C_param, G_param):
        # Physical constants and horizon scale
        rs = 2 * G_param * M_param / C_param**2
        
        # Effective charge-like term from the unified coupling
        # In this model, the coupling mediates a Reissner-Nordstrom-like correction
        rq_sq = (self.coupling * G_param) / C_param**2
        
        # Quantum regularization factor (prevents divergence at r -> 0)
        # Inspired by non-commutative geometry or loop quantum gravity corrections
        quantum_correction = torch.exp(-(self.ell**2) / (r**2 + 1e-40))
        
        # Modified shift function incorporating quantum gravity damping
        f_r = 1 - (rs / r) + (rq_sq / r**2) * quantum_correction
        
        g_tt = -f_r
        g_rr = 1/f_r
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp