import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    category = "modified_gravity"

    def __init__(self, beta=1e-10):
        super().__init__(name="Custom Modified Gravity Low Acceleration")
        self.beta = beta
        
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        """
        Lagrangian formulation:
        
        This modified gravity theory aims to strengthen the gravitational force at low accelerations,
        inspired by MOND-like phenomenology. The modification is introduced by altering the
        time-time component of the metric tensor.
        
        The standard Schwarzschild metric is given by:
        ds^2 = -(1 - rs/r) c^2 dt^2 + (1 - rs/r)^{-1} dr^2 + r^2 (dtheta^2 + sin^2(theta) dphi^2)
        where rs = 2GM/c^2 is the Schwarzschild radius.
        
        In the weak-field, slow-motion limit, the time-time component g_tt is related to the Newtonian potential Phi:
        g_tt approx -(1 + 2*Phi/c^2)
        For Schwarzschild, Phi approx -GM/r.
        
        This modified theory introduces a correction term to g_tt that becomes significant at large radii (low accelerations).
        The proposed modification to the function f(r) = 1 - rs/r is:
        f_modified(r) = 1 - rs/r + beta * sqrt(rs/r)
        
        where 'beta' is a model parameter, and sqrt(rs/r) is chosen because it decays slower than 1/r as r increases,
        thereby increasing the effective gravitational pull at large distances.
        
        The metric components are then derived from this modified function:
        g_tt = -(1 - rs/r + beta * sqrt(rs/r))
        g_rr = 1 / (1 - rs/r + beta * sqrt(rs/r))
        g_pp = r^2 (for spherical symmetry)
        g_tp = 0 (for static spacetime)
        
        The effective potential, derived from g_tt = -(1 + 2*Phi_eff/c^2), becomes:
        Phi_eff approx -(GM/r + beta * (GM/r)**(1/2) * r**(1/2))
        The force F = -dPhi_eff/dr will thus have an additional term that decays slower than 1/r^2,
        leading to stronger gravity at low accelerations.
        
        The Lagrangian density for a scalar field phi in this spacetime would be:
        L = (1/2) g_mu_nu * partial_mu(phi) * partial_nu(phi) - V(phi)
        
        The geodesic equation for a test particle is derived from:
        (1/2) g_mu_nu * dx^mu/dtau * dx^nu/dtau = constant
        
        The specific form of the modification aims to address the "Quantum Geodesic Sim" and "g-2 Muon" failures
        by altering the gravitational pull in regimes where standard gravity might deviate from observations.
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # The correction term is designed to be significant at large r (low acceleration).
        # A term proportional to sqrt(rs/r) decays slower than 1/r, strengthening gravity.
        correction_term = self.beta * torch.sqrt(rs / r)
        
        # The function f represents the deviation from flat spacetime.
        # f = 1 - rs/r is for Schwarzschild.
        # We modify it to f = 1 - rs/r + beta * sqrt(rs/r).
        f = 1 - rs/r + correction_term
        
        # Ensure f is positive to avoid issues with g_rr and potential singularities.
        # Clipping at a small positive epsilon is a common regularization technique.
        epsilon = 1e-10
        f = torch.maximum(f, torch.tensor(epsilon, device=r.device))
        
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2  # For spherical symmetry
        g_tp = torch.zeros_like(r) # For static spacetime
        
        return g_tt, g_rr, g_pp, g_tp