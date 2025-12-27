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
        Lagrangian formulation of the theory:
        This theory is a modification of General Relativity aiming to explain phenomena at low accelerations,
        inspired by MOND-like phenomenology. The modification is introduced by altering the time-time
        component of the metric, effectively modifying the gravitational potential.

        The underlying principle is to enhance gravitational effects at large distances (low accelerations)
        while remaining consistent with Newtonian gravity at high accelerations (small distances).

        The proposed metric modification is inspired by theories that introduce a length scale, `L`,
        such that at distances `r >> L`, the effective gravitational force is stronger than Newtonian.
        A common approach is to modify the effective potential. For a static, spherically symmetric
        spacetime, the metric is:
        ds^2 = -f(r) dt^2 + dr^2/f(r) + r^2 (dtheta^2 + sin^2(theta) dphi^2)
        where f(r) = 1 - 2GM/r for Schwarzschild.

        We modify f(r) to f_mod(r) = 1 - 2GM/r + \delta(r), where \delta(r) is a correction term.
        For low accelerations (large r), we want the effective force to be stronger.
        The force is related to the derivative of the potential, which is related to f(r).
        A term that decays slower than 1/r is needed.

        The chosen modification is:
        f_mod(r) = 1 - rs/r + beta * sqrt(rs/r)
        where rs = 2*G*M/C^2 is the Schwarzschild radius and beta is a new parameter.

        At large r, rs/r << 1 and sqrt(rs/r) << 1.
        The correction term beta * sqrt(rs/r) decays slower than rs/r.
        This leads to a stronger effective gravitational pull at large distances.

        The Lagrangian for a scalar field theory can be generalized to metric theories.
        For a scalar field phi, the action is S = integral(sqrt(-g) * L_scalar(phi, partial_mu phi) d^4x).
        In Einstein-Hilbert gravity, L_scalar is related to the Ricci scalar R.
        In modified gravity, the Lagrangian might be modified, for instance, f(R) gravity where
        S = integral(sqrt(-g) * f(R) d^4x).

        Alternatively, one can consider modifying the Einstein-Hilbert action directly by
        including scalar fields or modifying the gravitational action itself.
        A common approach in scalar-tensor theories is:
        S = integral(sqrt(-g) * [ G(phi) * R - W(phi) * g^{mu nu} partial_mu phi partial_nu phi ] d^4x)
        where G(phi) and W(phi) are functions of a scalar field.

        This specific implementation modifies the metric components directly, which can be
        thought of as arising from a specific scalar-tensor theory or a higher-order gravity theory.
        The effective Lagrangian governing the metric components would implicitly contain
        terms that lead to this modified f(r).

        For the purpose of this implementation, the direct modification of the metric components
        is sufficient. The choice of `beta * sqrt(rs/r)` for the correction term is heuristic,
        aiming to introduce a deviation from Newtonian gravity at large scales, consistent with
        phenomenological requirements of theories like MOND.

        The metric components are:
        g_tt = -(1 - rs/r + beta * sqrt(rs/r))
        g_rr = 1 / (1 - rs/r + beta * sqrt(rs/r))
        g_pp = r^2 (for spherical symmetry)
        g_tp = 0 (for static spacetime)

        The parameter `beta` controls the strength of the deviation from Newtonian gravity.
        A small `beta` ensures that at small `r` (high accelerations), the theory approaches
        the Schwarzschild solution.
        """
        rs = 2 * G_param * M_param / C_param**2

        # Ensure r is not zero or negative to avoid division by zero or sqrt of negative
        r_safe = torch.where(r <= 0, torch.tensor(1e-10, device=r.device), r)

        # Correction term that strengthens gravity at low accelerations (large r)
        # The term is proportional to sqrt(rs/r), which decays slower than 1/r.
        # This makes the effective gravitational pull stronger at larger radii.
        correction_term = self.beta * torch.sqrt(rs / r_safe)

        # The function f(r) for the metric components
        # f(r) = 1 - rs/r + correction_term
        f = 1 - rs / r_safe + correction_term

        # Ensure f is positive to avoid issues with g_rr (which involves 1/f).
        # A small positive epsilon is used to prevent division by zero or very small numbers.
        # This also helps in cases where the correction might push f to zero or negative values
        # for certain r, which would lead to singularities or unphysical metrics.
        epsilon = 1e-10
        f = torch.maximum(f, torch.tensor(epsilon, device=r.device))

        g_tt = -f
        g_rr = 1 / f
        g_pp = r_safe**2  # Spherical symmetry
        g_tp = torch.zeros_like(r) # Static spacetime

        return g_tt, g_rr, g_pp, g_tp