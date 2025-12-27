import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    category = "modified_gravity"

    def __init__(self, beta=1e-10):
        super().__init__(name="Custom Modified Gravity")
        self.beta = beta

    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2

        # This modification aims to strengthen gravity at low accelerations (large r)
        # by adding a term to the Schwarzschild potential that decays slower than 1/r.
        # The chosen form is beta * sqrt(rs/r), which leads to a correction
        # in the effective force that decays as r^(-3/2).

        # We modify the function f(r) in the Schwarzschild metric g_tt = -f, g_rr = 1/f.
        # The standard Schwarzschild has f = 1 - rs/r.
        # Our modified f is f = 1 - rs/r + beta * sqrt(rs/r).

        # This form is inspired by theories that exhibit a crossover acceleration.
        # At large r (low acceleration), the beta term becomes significant,
        # increasing the effective gravitational pull.

        # The choice of beta influences the strength of the modification.
        # A small beta ensures that at small radii (high acceleration),
        # the metric closely resembles Schwarzschild.

        # To ensure the metric is well-behaved (g_rr is positive), the denominator
        # 1 - rs/r + beta * sqrt(rs/r) must be positive.
        # Let x = sqrt(rs/r). The condition becomes 1 - x^2 + beta*x > 0.
        # This quadratic in x is positive for x within its roots.
        # The relevant positive root is approximately 1 - beta/2 for small beta.
        # Thus, we need sqrt(rs/r) < 1 - beta/2, which means r > rs / (1 - beta/2)^2.
        # This implies the modification is significant at large radii, as intended.

        # We add a small epsilon to prevent division by zero or negative values
        # in the denominator of g_rr, ensuring numerical stability.
        epsilon = 1e-10
        
        # Calculate the term inside the square root, ensuring it's non-negative.
        sqrt_term_arg = torch.clamp(rs / r, min=0.0)
        correction_term = self.beta * torch.sqrt(sqrt_term_arg)
        
        f_val = 1 - rs / r + correction_term
        f = torch.maximum(f_val, torch.tensor(epsilon, device=r.device))
        
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp

    def lagrangian(self, **kwargs):
        """
        Lagrangian formulation for the Modified Gravity Low Acceleration theory.

        This theory modifies the spacetime geometry to account for observed gravitational
        effects at low accelerations, potentially addressing phenomena like galactic
        rotation curves without invoking dark matter. The modification is introduced
        by altering the metric components, specifically g_tt and g_rr, which
        effectively modifies the gravitational potential.

        In the weak-field, slow-motion limit, the gravitational potential Phi is related
        to the time-time component of the metric by g_tt ~ -(1 + 2*Phi/c^2).
        The standard Schwarzschild metric gives Phi ~ -GM/r.
        Our modification introduces a term that grows slower than 1/r at large distances,
        effectively increasing the gravitational pull where accelerations are small.

        The Lagrangian for a scalar field phi in a curved spacetime is given by:
        L = (1/2) g^{mu nu} partial_mu phi partial_nu phi - V(phi)

        For a gravitational theory, we can consider the Einstein-Hilbert action:
        S = (1 / (16 * pi * G)) integral (R - 2*Lambda) sqrt(-g) d^4x
        where R is the Ricci scalar and Lambda is the cosmological constant.

        In a modified gravity framework, the action might be altered, for example,
        in f(R) gravity, the action is S = integral f(R) sqrt(-g) d^4x.

        This specific implementation modifies the metric directly, implying a departure
        from standard GR, potentially within a scalar-tensor theory or a more general
        geometric modification. The effective Lagrangian for a test particle moving
        along a geodesic is related to the metric:

        L_particle = (1/2) m * g_{mu nu} (dx^mu/dtau) (dx^nu/dtau)

        The modification to g_tt and g_rr directly impacts the geodesic equation and
        thus the motion of test particles. The specific form of the modification,
        g_tt = -(1 - rs/r + beta * sqrt(rs/r)) and g_rr = 1 / (1 - rs/r + beta * sqrt(rs/r)),
        changes the effective gravitational potential and force law at large distances.

        The modification can be thought of as introducing an effective potential term
        that is more attractive than Newtonian gravity at low accelerations.
        The parameter beta controls the strength of this deviation from GR.
        """
        # This method is intended to return a symbolic representation of the Lagrangian.
        # For this specific metric modification, deriving a closed-form Lagrangian
        # can be complex and depends on the underlying theoretical framework (e.g.,
        # scalar-tensor theory, f(R) gravity, etc.).
        # For now, we return a placeholder or a simplified representation if applicable.

        # Placeholder for Lagrangian representation.
        # A full derivation would require specifying the underlying field content and action.
        # For a metric-based modification like this, the Lagrangian of matter fields
        # will be affected via the metric. The gravitational part of the Lagrangian
        # itself is what is being modified.
        
        # Let's define symbolic variables for convenience
        r, M, C, G, beta_sym = sp.symbols('r M C G beta')
        rs_sym = 2 * G * M / C**2
        
        # Symbolic representation of the modified metric components
        f_sym = 1 - rs_sym / r + beta_sym * sp.sqrt(rs_sym / r)
        
        # We can represent the modified Einstein-Hilbert action conceptually.
        # In this modified theory, the Ricci scalar R would be computed from the modified metric.
        # The action S = integral (R - 2*Lambda) sqrt(-g) d^4x would then follow.
        # However, explicitly calculating R for this metric and providing a full symbolic Lagrangian
        # is computationally intensive and beyond a simple placeholder.
        
        # A simplified Lagrangian for a test particle in this metric:
        # L_particle = (1/2) * ( -f * (dt/dtau)^2 + (1/f) * (dr/dtau)^2 + r^2 * (dtheta/dtau)^2 + r^2 * sin^2(theta) * (dphi/dtau)^2 )
        # This is not the gravitational Lagrangian itself, but the matter Lagrangian in the curved spacetime.
        
        # For the purpose of this placeholder, we can represent the modified part of the action.
        # The modification is embedded in the metric components derived from the theory's field equations.
        # A common approach for modified gravity is to postulate an action.
        # For instance, if this arose from a scalar-tensor theory, the action would be:
        # S = integral ( Phi * R - 2 * V(Phi) ) sqrt(-g) d^4x + S_matter(g_mu nu, psi)
        # Where Phi is a scalar field and V(Phi) is its potential.
        
        # Given the direct metric modification, a full Lagrangian derivation is complex.
        # We will return a symbolic representation of the metric function 'f' which defines the theory's geometry.
        
        return {
            "metric_f_symbolic": (1 - rs_sym / r + beta_sym * sp.sqrt(rs_sym / r)),
            "rs_symbolic": rs_sym,
            "description": "Symbolic representation of the modified metric function f(r) and Schwarzschild radius rs."
        }