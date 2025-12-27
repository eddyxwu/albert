import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Exponentially Modified Schwarzschild Metric (Yukawa-like).

    This theory proposes a modification to the standard Schwarzschild metric by introducing an
    exponentially decaying correction term. Such a correction is characteristic of theories
    where gravity is mediated by massive particles (e.g., massive gravitons) or where a new
    fundamental scalar field couples to matter, leading to a "fifth force" that is
    Yukawa-suppressed at large distances.

    A prominent example is massive gravity, where the graviton acquires a small mass,
    leading to a modified gravitational potential that falls off faster than 1/r.
    Alternatively, in screened modified gravity models (e.g., chameleon, symmetron, dilaton
    theories), a scalar field mediates a new force which is typically suppressed in dense
    environments but can manifest as an exponential deviation from General Relativity
    in low-density regions or at specific distance scales.

    A potential Lagrangian formulation that could lead to such a metric modification
    might involve adding a mass term for the graviton in the gravitational action or
    introducing a new scalar field with specific potential and coupling terms to matter
    and curvature. For instance, in a massive gravity theory, the action might include
    terms like:

    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} (R - m_g^2 U(g_{\mu\nu})) + L_{matter} \right]

    where U(g_{\mu\nu}) is a potential term for the graviton mass m_g. In scalar-tensor
    theories, the action might be:

    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} (R - \omega(\phi) (\nabla\phi)^2 - V(\phi)) + L_{matter}(\phi) \right]

    The parameters \alpha and \lambda_scale in the metric correction term (\alpha * exp(-r/\lambda_scale)) are
    phenomenological coefficients. \alpha represents the strength of the correction, and
    \lambda_scale is the characteristic length scale of the exponential decay, often related
    to the Compton wavelength of the massive graviton (1/m_g) or the screening length of a
    scalar field. This metric represents a static, spherically symmetric vacuum solution
    in such a modified gravity theory.
    """
    category = "modified_gravity"
    
    def __init__(self, alpha=0.01, lambda_scale=1e10): 
        super().__init__(name="ExponentialCustomTheory", force_6dof_solver=False)
        self.alpha = alpha  # Strength of the exponential correction
        self.lambda_scale = lambda_scale # Characteristic decay length of the exponential correction (e.g., in meters)
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # Exponential correction term: alpha * exp(-r/lambda_scale)
        # Ensure lambda_scale is positive to avoid division by zero or non-physical behavior
        safe_lambda_scale = torch.maximum(self.lambda_scale, torch.tensor(1e-10, device=r.device))
        
        correction = self.alpha * torch.exp(-r / safe_lambda_scale)
        
        f = 1 - rs/r + correction
        
        # Ensure 'f' does not become zero or negative to avoid division by zero or imaginary metrics
        f = torch.maximum(f, torch.tensor(1e-10, device=r.device))
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r**2 # Represents the angular component g_theta_theta * sin^2(theta) or g_phi_phi
        g_tp = torch.zeros_like(r) # For a static, non-rotating metric
        
        return g_tt, g_rr, g_pp, g_tp