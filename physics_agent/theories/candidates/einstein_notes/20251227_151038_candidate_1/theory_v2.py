import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    category = "modified_gravity"

    def __init__(self, beta=1e-10):
        super().__init__(name="Modified Gravity Low Acceleration")
        self.beta = beta
        
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        """
        Lagrangian formulation:
        The theory is a modification of General Relativity, specifically targeting the behavior at low accelerations.
        This is achieved by modifying the metric components.
        
        Consider a modified Einstein-Hilbert action:
        S = 1/(16*pi*G) * integral( (R - L_m) * sqrt(-g) d^4x )
        
        In the context of modified gravity theories, modifications can arise from:
        1. Adding higher-order curvature invariants to the Lagrangian (e.g., f(R) gravity).
        2. Introducing scalar fields coupled to gravity.
        3. Modifying the gravitational coupling constant or introducing a position-dependent coupling.
        
        This implementation directly modifies the metric components, which can be thought of as arising from a modified
        stress-energy tensor or a modified gravitational action.
        
        The modification aims to increase the effective gravitational pull at large distances (low accelerations)
        compared to standard Schwarzschild. This is achieved by altering the time-time component of the metric.
        
        The standard Schwarzschild metric is:
        ds^2 = -(1 - rs/r) c^2 dt^2 + dr^2/(1 - rs/r) + r^2 (d_theta^2 + sin^2_theta d_phi^2)
        where rs = 2GM/c^2.
        
        The proposed modification to the time-time component g_tt is:
        g_tt = -(1 - rs/r + beta * sqrt(rs/r))
        
        This leads to a modification in the effective gravitational potential.
        The effective potential Phi is related by g_tt = -(1 + 2*Phi/c^2).
        So, Phi approx -(rs/r + beta * sqrt(rs/r)) * c^2 / 2.
        
        The force F = -dPhi/dr.
        F approx (rs/r^2 + (beta/2) * sqrt(rs) * r**(-3/2)) * c^2 / 2.
        
        The second term (beta * sqrt(rs) * r**(-3/2)) decays slower than 1/r^2,
        leading to stronger gravity at large r (low acceleration).
        
        The Lagrangian density for this modified gravity can be complex and depends on how the modification is derived.
        If we consider this as a direct modification of spacetime geometry without introducing new fields,
        the underlying action might be a modification of the Einstein-Hilbert action, or it could be
        interpreted as an effective description arising from a more fundamental theory.
        
        A possible way to achieve this metric form might be through a scalar-tensor theory where the scalar field
        couples in a specific way to gravity, or through an f(R) theory that, in the static spherically symmetric case,
        yields such a metric. However, without a full derivation from a fundamental action, this metric
        is treated as an ansatz.
        
        For the purpose of this class, we focus on the metric components derived from the ansatz.
        The 'beta' parameter controls the strength of the modification, and its units are such that
        beta * sqrt(rs/r) is dimensionless. Given rs has units of length, beta must also have units of length^(1/2).
        However, in the code, beta is treated as a dimensionless coupling constant, and rs/r is dimensionless.
        The effective dimensional analysis implies beta should have units of length^(1/2) if rs is length.
        But since beta is given as a small dimensionless number, the scaling is implicitly handled.
        
        The parameter C_param is related to the speed of light, and G_param is the gravitational constant.
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # The correction term is designed to be significant at large r (low acceleration).
        # The term sqrt(rs/r) decays slower than 1/r, making the overall gravitational pull stronger.
        correction_term = self.beta * torch.sqrt(rs / r)
        
        # The function f represents the deviation from flat spacetime in the (1 - rs/r) term.
        # We need f to remain positive for g_rr to be well-defined and positive.
        # The condition for f > 0 is r > rs / (1 - beta/2)^2 for small beta.
        # This ensures the metric is regular in the region of interest for low acceleration modification.
        f = 1 - rs/r + correction_term
        
        # Ensure f is positive to avoid issues with g_rr.
        # Clip f to a small positive value if it goes negative, though for typical small beta and r > rs, this should be positive.
        epsilon = 1e-10  # A small positive number to prevent division by zero or negative values.
        f = torch.maximum(f, torch.tensor(epsilon, device=r.device))
        
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2  # This assumes spherical symmetry and is the angular part of the metric.
        g_tp = torch.zeros_like(r) # Assuming static spacetime, so no t-phi, t-theta, etc. couplings.
        
        return g_tt, g_rr, g_pp, g_tp