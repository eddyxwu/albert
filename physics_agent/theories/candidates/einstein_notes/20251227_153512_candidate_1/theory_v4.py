import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Logarithmically Modified Schwarzschild Metric.

    This theory proposes a modification to the standard Schwarzschild metric by introducing a
    logarithmic correction term. Such corrections can arise from various extensions of General
    Relativity, particularly theories involving quantum gravitational effects or specific
    higher-order curvature invariants that lead to scale-dependent couplings (e.g., Renormalization Group improved gravity).
    Logarithmic terms can also appear in certain string theory compactifications or in effective
    field theories beyond General Relativity.

    A potential Lagrangian formulation that could lead to such a metric modification
    might involve an action of the form:

    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} (R + \text{higher-order curvature terms} + \text{logarithmic terms}) + L_{matter} \right]

    where the logarithmic terms could arise from loop corrections in quantum gravity or from
    specific non-minimal couplings. For example, some models might involve a running gravitational
    constant G that effectively introduces such terms, or specific effective potentials derived from
    brane-world models at certain scales.

    The parameter \alpha in the metric correction term (\alpha \frac{\log(r/rs)}{r}) is a
    phenomenological coefficient related to the strength of these quantum or modified gravitational
    effects. This metric represents a static, spherically symmetric vacuum solution in such a
    modified gravity theory.
    """
    category = "modified_gravity"
    
    def __init__(self, alpha=0.01): 
        super().__init__(name="LogarithmicCustomTheory", force_6dof_solver=False)
        self.alpha = alpha  # Strength of the logarithmic correction
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # Logarithmic correction term: alpha * log(r/rs) / r
        # Ensure r/rs is positive for the logarithm
        # Also ensure r is not zero in the denominator
        r_safe = torch.maximum(r, torch.tensor(1e-10, device=r.device))
        ratio_r_rs = r_safe / rs
        
        correction = self.alpha * torch.log(ratio_r_rs) / r_safe
        
        f = 1 - rs/r_safe + correction
        
        # Ensure 'f' does not become zero or negative to avoid division by zero or imaginary metrics
        f = torch.maximum(f, torch.tensor(1e-10, device=r.device))
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r_safe**2 # Represents the angular component g_theta_theta * sin^2(theta) or g_phi_phi
        g_tp = torch.zeros_like(r) # For a static, non-rotating metric
        
        return g_tt, g_rr, g_pp, g_tp