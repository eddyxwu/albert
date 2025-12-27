import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    category = "modified_gravity"

    def __init__(self, beta=1e-10):
        super().__init__(name="CustomTheory")
        self.beta = beta
        
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        """
        Lagrangian formulation of CustomTheory:

        This modified gravity theory aims to enhance gravitational attraction at low accelerations,
        inspired by MOND-like phenomenology. The modification is introduced by altering the
        time-time component of the metric, g_tt.

        The standard Schwarzschild metric in spherical coordinates is given by:
        ds^2 = -(1 - rs/r) dt^2 + (1 - rs/r)^-1 dr^2 + r^2 (dtheta^2 + sin^2(theta) dphi^2)
        where rs = 2GM/c^2 is the Schwarzschild radius.

        In this modified theory, we adjust g_tt to:
        g_tt = -(1 - rs/r + beta * sqrt(rs/r))

        This leads to the modified metric components:
        g_tt = -(1 - rs/r + beta * sqrt(rs/r))
        g_rr = 1 / (1 - rs/r + beta * sqrt(rs/r))
        g_pp = r^2 (where pp represents the angular part, dtheta^2 + sin^2(theta) dphi^2)
        g_tp = 0 (for the off-diagonal components)

        The term `beta * sqrt(rs/r)` acts as a correction that becomes more significant at larger radii (lower accelerations),
        effectively strengthening the gravitational pull compared to standard Newtonian gravity at these scales.
        The parameter `beta` controls the strength of this modification.

        The underlying principle is to modify the effective gravitational potential such that its derivative (the force)
        is larger than the Newtonian prediction for accelerations below a certain characteristic scale.
        The term `beta * sqrt(rs/r)` decays slower than `rs/r` as `r` increases, achieving this effect.

        To ensure the metric is well-behaved and physically meaningful (especially for g_rr), the denominator
        `f = 1 - rs/r + beta * sqrt(rs/r)` must remain positive. This condition imposes a constraint on `r`
        such that `r > rs / (1 - beta/2)^2` for a small `beta`. This is consistent with the idea that
        the modification primarily affects large-scale gravitational phenomena.

        The Lagrangian for a scalar field phi in curved spacetime is typically given by:
        L = (1/2) g^munu nabla_mu phi nabla_nu phi - V(phi)
        where g^munu is the inverse metric. The specific form of the action and potential V(phi) would depend
        on the precise particle content and dynamics being described. For purely gravitational effects,
        one might consider the Einstein-Hilbert action modified by curvature terms, but this metric
        modification can be seen as a phenomenological approach to capture certain gravitational effects
        without necessarily deriving it from a fundamental action principle in this context.

        For the purpose of this implementation, we focus on providing the metric components consistent
        with the described modification.
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # The modification to g_tt is designed to increase effective gravity at low accelerations (large r).
        # The term beta * sqrt(rs/r) decays slower than 1/r as r increases.
        # This makes the effective potential deeper at large distances.
        
        # Define the common factor f = 1 - rs/r + beta * sqrt(rs/r)
        # This factor appears in both g_tt and g_rr.
        # We need f > 0 for g_rr to be well-defined and positive.
        
        # Ensure r is not zero to avoid division by zero.
        # In physical scenarios, r is typically > 0.
        # Add a small epsilon to r if it's exactly zero, or handle as a special case if needed.
        # For this implementation, assuming r > 0 from the context of metric components.

        # Calculate the correction term.
        # Use torch.sqrt for element-wise square root.
        correction_term = self.beta * torch.sqrt(rs / r)
        
        # Combine terms for f.
        f = 1.0 - rs / r + correction_term
        
        # To avoid numerical issues and ensure positivity of f, especially for small r or large beta,
        # we can clip f to a small positive value. This is a common practice when dealing with
        # potentially pathological regions of spacetime or parameter space.
        # The condition for f > 0 derived in comments suggests a minimum r based on beta.
        # Clipping provides a more robust numerical solution.
        epsilon = 1e-12  # A small positive number to ensure f is not zero or negative.
        f = torch.maximum(f, torch.tensor(epsilon, device=r.device, dtype=r.dtype))
        
        # Metric components:
        g_tt = -f
        g_rr = 1.0 / f
        g_pp = r**2  # This represents the angular part: r^2 (dtheta^2 + sin^2(theta) dphi^2)
        g_tp = torch.zeros_like(r) # Off-diagonal component
        
        return g_tt, g_rr, g_pp, g_tp