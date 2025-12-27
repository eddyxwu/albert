import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Exponentially Modified Schwarzschild Metric.

    This theory proposes a modification to the standard Schwarzschild metric by introducing an
    exponentially decaying correction term. This type of correction can phenomenologically
    represent deviations from General Relativity at certain scales, or it can arise from
    extended theories of gravity, such as those involving a massive scalar field.

    A potential Lagrangian formulation that could lead to such a metric modification in
    a static, spherically symmetric spacetime, particularly if the exponential term
    is sourced by a massive scalar field \phi, is an action of the form:

    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} R - \frac{1}{2} g^{\mu\nu} \partial_\mu \phi \partial_\nu \phi - \frac{1}{2} m^2 \phi^2 + L_{matter} \right]

    where R is the Ricci scalar, G is Newton's gravitational constant, \phi is a scalar field
    with mass m, and L_{matter} is the Lagrangian for ordinary matter.
    In this context, the parameter \lambda_scale in the metric correction term
    (\alpha e^{-r/\lambda_scale}) is related to the Compton wavelength of the scalar field,
    i.e., \lambda_scale \sim \hbar/(mc). The parameter \alpha quantifies the strength
    of this exponential correction.
    The metric presented here is a specific solution to the field equations derived from such
    an action, or a similar modified gravity theory, in the vacuum outside a spherical mass.
    """
    category = "modified_gravity"
    
    def __init__(self, alpha=0.01, lambda_scale=1e5): 
        super().__init__(name="ExponentialCustomTheory", force_6dof_solver=False)
        self.alpha = alpha  # Strength of the exponential correction
        self.lambda_scale = lambda_scale # Characteristic decay length of the exponential correction
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # Exponential correction term
        correction = self.alpha * torch.exp(-r / self.lambda_scale)
        
        f = 1 - rs/r + correction
        
        # Ensure 'f' does not become zero or negative
        f = torch.maximum(f, torch.tensor(1e-10, device=r.device))
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r**2 
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp