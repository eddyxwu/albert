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
        The theory is a modification of General Relativity, specifically targeting the weak field limit.
        The modification aims to increase gravitational attraction at low accelerations,
        potentially explaining phenomena like galactic rotation curves without dark matter.

        In the weak field approximation, the spacetime interval ds^2 is given by:
        ds^2 = -(1 + 2*Phi/c^2)c^2 dt^2 + (1 - 2*Phi/c^2)(dx^2 + dy^2 + dz^2)

        For a static, spherically symmetric metric like Schwarzschild, this simplifies to:
        ds^2 = -f(r) c^2 dt^2 + (1/f(r)) dr^2 + r^2 (dtheta^2 + sin^2(theta) dphi^2)
        where f(r) = 1 - rs/r for Schwarzschild.

        The modification is applied to the f(r) term in the metric.
        Instead of f(r) = 1 - rs/r, we use:
        f(r) = 1 - rs/r + beta * sqrt(rs/r)

        This modification introduces a term that decays slower than 1/r at large distances,
        effectively strengthening gravity in the low acceleration regime.

        The Lagrangian density for General Relativity is L_GR = sqrt(-g) * R / (16*pi*G),
        where R is the Ricci scalar.
        Modifications to gravity can be formulated by changing the Lagrangian, for instance,
        in f(R) gravity, L = sqrt(-g) * f(R) / (16*pi*G).

        This specific modification can be viewed as a phenomenological approach to alter the
        effective gravitational potential without explicitly defining a modified Ricci scalar.
        The effective potential Phi_eff can be derived from the modified metric component g_tt:
        g_tt = -f(r) c^2 = -(1 - rs/r + beta * sqrt(rs/r)) c^2
        Comparing with ds^2 = -(1 + 2*Phi_eff/c^2)c^2 dt^2, we get:
        1 + 2*Phi_eff/c^2 = 1 - rs/r + beta * sqrt(rs/r)
        Phi_eff = (C_param**2 / 2) * (-rs/r + beta * sqrt(rs/r))
        where C_param is the speed of light.

        The force F_eff = -d(Phi_eff)/dr.
        F_eff = -(C_param**2 / 2) * (-rs/r^2 + beta * (1/2) * sqrt(rs) * r**(-3/2))
        F_eff = (C_param**2 / 2) * (rs/r^2 - (beta/2) * sqrt(rs) * r**(-3/2))

        At large r, the Newtonian force is proportional to rs/r^2.
        The modification adds a term proportional to r**(-3/2), which decays slower than 1/r^2,
        leading to stronger attraction at low accelerations (large r).

        The parameter beta controls the strength of this modification.
        The term sqrt(rs/r) is dimensionless, as is rs/r.
        The parameter beta is a dimensionless coupling constant for this modification.
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # The function f(r) that determines g_tt and g_rr
        # f(r) = 1 - rs/r + beta * sqrt(rs/r)
        f_r = 1 - rs / r + self.beta * torch.sqrt(rs / r)
        
        # Ensure f_r is positive to avoid issues with g_rr (which is 1/f_r)
        # and to maintain a well-behaved metric.
        # The condition for f_r > 0 is approximately r > rs / (1 - beta/2)^2 for small beta.
        # Clipping to a small positive epsilon handles potential numerical issues or pathological regions.
        epsilon = 1e-15  # Use a smaller epsilon for better precision
        f_r = torch.maximum(f_r, torch.tensor(epsilon, device=r.device))
        
        g_tt = -f_r
        g_rr = 1 / f_r
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp