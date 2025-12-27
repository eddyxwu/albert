import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Inverse-Square Modified Schwarzschild Metric.

    This theory proposes a modification to the standard Schwarzschild metric by introducing
    an inverse-square correction term. Such corrections can arise from various contexts
    in modified gravity, including:

    1.  **Non-minimal coupling of a scalar field to gravity:** In scalar-tensor theories,
        if a scalar field couples to the Ricci scalar (e.g., `f(\phi) R`), its vacuum
        expectation value or dynamics can generate effective potential terms in the metric,
        sometimes leading to `1/r^2` or similar corrections.
    2.  **Extra spatial dimensions:** In models with large extra dimensions, the
        gravitational potential can deviate from the `1/r` Newtonian law at short distances,
        potentially leading to inverse-square corrections.
    3.  **Phenomenological extensions to Newtonian gravity:** At very short ranges,
        laboratory tests often constrain deviations from the inverse-square law of gravity.
        This theory considers a direct modification to the metric function that effectively
        introduces an additional `1/r^2` potential.
    4.  **Effective Field Theories:** In certain effective field theory approaches to
        quantum gravity, higher-dimensional operators or integration of heavy fields
        could manifest as `1/r^n` corrections to the gravitational potential.

    A potential Lagrangian formulation that could lead to such a metric modification
    might involve an action of the form:

    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} R + \mathcal{L}_{matter} + \mathcal{L}_{mod} \right]

    where `\mathcal{L}_{mod}` could represent terms like:
    - Non-minimal coupling: `\xi \phi^2 R` where a scalar field `\phi` acquires a background value.
    - Higher-dimensional operators in the vacuum: `\Lambda^{-2} R_{\mu\nu\rho\sigma} R^{\mu\nu\rho\sigma}` or similar, which upon solving field equations in specific limits could yield such terms.
    - A direct modification to the gravitational action that, when linearized, yields an `1/r^2` potential.

    The parameter `alpha` governs the strength of this inverse-square modification,
    while `epsilon` is a regularization parameter to prevent divergence at `r=0` and
    can be interpreted as a characteristic length scale where these effects become dominant
    or where the classical approximation breaks down. This metric represents a static,
    spherically symmetric vacuum solution in such a modified gravity theory.
    """
    category = "modified_gravity"
    
    def __init__(self, alpha=0.01, epsilon=1e-12):
        super().__init__(name="InverseSquareModifiedSchwarzschild", force_6dof_solver=False)
        self.alpha = alpha      # Strength of the inverse-square correction
        self.epsilon = epsilon  # Regularization parameter for r^2 term
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # Ensure r is positive and not zero to avoid division by zero
        r_safe = torch.maximum(r, torch.tensor(1e-10, device=r.device))
        
        # Inverse-square correction term: alpha / (r^2 + epsilon)
        correction = self.alpha / (r_safe**2 + self.epsilon)
        
        f = 1 - rs / r_safe + correction
        
        # Ensure 'f' does not become zero or negative to avoid division by zero or imaginary metrics
        f = torch.maximum(f, torch.tensor(1e-10, device=r.device))
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r_safe**2 # Represents the angular component g_theta_theta * sin^2(theta) or g_phi_phi
        g_tp = torch.zeros_like(r) # For a static, non-rotating metric
        
        return g_tt, g_rr, g_pp, g_tp