import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    category = "modified_gravity"

    def __init__(self, beta=1e-10):
        super().__init__(name="Custom Theory")
        self.beta = beta
        
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        """
        Lagrangian formulation of the modified gravity theory.

        This theory modifies the Schwarzschild metric to incorporate a gravitational
        force that is stronger at low accelerations, inspired by MOND-like
        phenomenology. The modification is introduced by altering the time-time
        component of the metric.

        The action for this theory can be thought of as originating from a modified
        Einstein-Hilbert action, or equivalently, a modification to the gravitational
        potential. For a static, spherically symmetric spacetime, the metric components
        are:
        ds^2 = -f(r) dt^2 + (1/f(r)) dr^2 + r^2 (dtheta^2 + sin^2(theta) dphi^2)

        where f(r) is modified from the Schwarzschild case.

        In the Schwarzschild metric, f(r) = 1 - rs/r, where rs = 2GM/c^2.
        The effective gravitational potential is approximately Phi = -GM/r.
        The force is F = -dPhi/dr = -GM/r^2.

        In this modified theory, we propose:
        f(r) = 1 - rs/r + beta * sqrt(rs/r)

        where 'beta' is a parameter controlling the strength of the modification.
        The term beta * sqrt(rs/r) becomes significant at large r (low acceleration).

        The effective potential is approximately:
        Phi(r) approx - (GM/r + beta * sqrt(GM/r) * C_param^2 / 2)

        The force is then:
        F(r) = -dPhi/dr approx (GM/r^2 + (beta/2) * sqrt(GM) * r^(-3/2)) * C_param^2 / 2

        The second term indicates a stronger attraction than Newtonian gravity at large r,
        as r^(-3/2) decays slower than r^(-2).

        This modification aims to address observations of galaxy rotation curves without
        resorting to dark matter. The parameter 'beta' is related to the characteristic
        acceleration scale a0 in MOND.
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # The correction term is designed to increase the gravitational pull at large radii (low accelerations).
        # A term proportional to sqrt(rs/r) decays slower than 1/r, thus increasing the force at large r.
        correction_term = self.beta * torch.sqrt(rs / r)
        
        # The function f(r) determines the metric components.
        # For the metric to be well-behaved, f(r) should be positive.
        # The standard Schwarzschild metric has f(r) = 1 - rs/r.
        # Our modification is f(r) = 1 - rs/r + beta * sqrt(rs/r).
        # We need to ensure that 1 - rs/r + beta * sqrt(rs/r) > 0.
        # Let x = sqrt(rs/r). The condition becomes 1 - x^2 + beta*x > 0.
        # This quadratic in x has roots related to (beta +/- sqrt(beta^2 + 4))/2.
        # For positive x, the condition holds for x < (sqrt(beta^2 + 4) - beta)/2.
        # This implies r > rs / (((sqrt(beta^2 + 4) - beta)/2)^2).
        # For small beta, this is roughly r > rs / (1 - beta)^2.
        # This means the metric is valid for r outside a certain range, which is typical for modified gravity theories
        # aiming to explain large-scale phenomena.

        f = 1 - rs / r + correction_term
        
        # To prevent numerical issues where f might become non-positive for small r or large beta,
        # we enforce a minimum positive value.
        epsilon = 1e-12  # A small positive number
        f = torch.maximum(f, torch.tensor(epsilon, device=r.device, dtype=r.dtype))
        
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2  # For spherical symmetry, g_theta_theta = r^2, g_phi_phi = r^2 * sin^2(theta).
                     # For the purpose of calculating geodesic equations in a plane (theta=pi/2),
                     # we can use g_pp = r^2 representing the angular part.
        g_tp = torch.zeros_like(r) # Off-diagonal components are zero for static spacetime.
        
        return g_tt, g_rr, g_pp, g_tp