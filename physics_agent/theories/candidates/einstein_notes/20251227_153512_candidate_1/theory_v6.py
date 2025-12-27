import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Polynomial Modified Schwarzschild Metric.

    This theory proposes a modification to the standard Schwarzschild metric by introducing
    polynomial correction terms in powers of (rs/r). Such corrections are common in
    various extensions of General Relativity, particularly in the post-Newtonian (PN)
    expansion of metric theories of gravity, or in theories derived from higher-order
    curvature invariants. For example, f(R) gravity models, Gauss-Bonnet gravity, or
    other higher-derivative theories often yield such polynomial corrections in the weak-field
    and strong-field regimes. These terms can be interpreted as effective corrections due to
    the underlying quantum gravitational effects or classical deviations from GR at specific scales.

    A potential Lagrangian formulation that could lead to such a metric modification
    might involve an action of the form:

    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} (R + \ell_1 R^2 + \ell_2 R_{\mu\nu}R^{\mu\nu} + \dots) + L_{matter} \right]

    where \ell_1, \ell_2 are coupling constants for higher-order curvature invariants.
    When solving the field equations for a static, spherically symmetric vacuum,
    these higher-order terms can effectively generate corrections to the Schwarzschild
    metric that appear as powers of (rs/r).

    The parameters \alpha and \beta are phenomenological coefficients related to the
    strength of these modified gravitational effects. This metric represents a static,
    spherically symmetric vacuum solution in such a modified gravity theory.
    """
    category = "modified_gravity"
    
    def __init__(self, alpha=0.01, beta=0.001):
        super().__init__(name="PolynomialCustomTheory", force_6dof_solver=False)
        self.alpha = alpha  # Strength of the first polynomial correction
        self.beta = beta    # Strength of the second polynomial correction
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # Ensure r is positive and not zero to avoid division by zero
        r_safe = torch.maximum(r, torch.tensor(1e-10, device=r.device))
        
        # Polynomial correction terms: alpha * (rs/r) + beta * (rs/r)^2
        # Note: The original example had alpha * log(r/rs) / r.
        # This new correction is alpha * (rs/r) and beta * (rs/r)^2.
        # This is a different mathematical form as requested.
        
        rs_over_r = rs / r_safe
        correction = self.alpha * rs_over_r + self.beta * (rs_over_r**2)
        
        f = 1 - rs_over_r + correction
        
        # Ensure 'f' does not become zero or negative to avoid division by zero or imaginary metrics
        f = torch.maximum(f, torch.tensor(1e-10, device=r.device))
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r_safe**2 # Represents the angular component g_theta_theta * sin^2(theta) or g_phi_phi
        g_tp = torch.zeros_like(r) # For a static, non-rotating metric
        
        return g_tt, g_rr, g_pp, g_tp