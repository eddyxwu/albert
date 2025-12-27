import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    This theory extends the Schwarzschild metric with an exponential correction term.
    Such corrections frequently arise in various modified gravity theories,
    including scalar-tensor theories, f(R) gravity, or models incorporating extra dimensions,
    where a Yukawa-like interaction might modify the effective gravitational potential.

    The metric is assumed to be a static, spherically symmetric solution, given by:
    ds^2 = -f(r) c^2 dt^2 + f(r)^-1 dr^2 + r^2 (dtheta^2 + sin^2(theta) dphi^2)
    where the modified potential function f(r) is defined as:
    f(r) = 1 - rs/r + alpha * exp(-r/lambda_scale)
    and rs = 2GM/c^2 is the Schwarzschild radius.

    The additional term, `alpha * exp(-r/lambda_scale)`, represents a deviation
    from pure General Relativity's Schwarzschild solution. This exponential form
    is characteristic of a massive mediator field (like a massive scalar field)
    that generates a Yukawa potential.

    **Conceptual Lagrangian Formulation:**
    This metric modification conceptually stems from a theory of gravity that
    differs from the Einstein-Hilbert action alone. For instance, in a
    scalar-tensor theory, the action might take the form:

    S = \int d^4x \sqrt{-g} [ R/(16\pi G) - 1/2 g^{\mu\nu} \partial_\mu\phi \partial_\nu\phi - V(\phi) ] + S_matter

    where R is the Ricci scalar, G is Newton's gravitational constant, \phi is a
    scalar field, and V(\phi) is its potential. Appropriate choices for the scalar
    field potential V(\phi) or its coupling to matter fields (not explicitly shown
    in the vacuum metric here) can lead to solutions where the metric components
    include exponential deviations from the Schwarzschild form, especially in the
    weak field limit or as exact solutions in certain strong field regimes.
    The parameter 'lambda_scale' would typically relate to the Compton wavelength
    of the mediating scalar field, and 'alpha' would relate to its coupling strength.
    Alternatively, this could also arise from modifications to the gravitational
    Lagrangian itself, such as in f(R) gravity, where the Ricci scalar R in the
    action is replaced by an arbitrary function f(R).
    """
    
    def __init__(self):
        super().__init__(name="CustomTheory", force_6dof_solver=False)
        self.category = "classical"
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        # Extract parameters for the exponential correction from kwargs
        # Default values are provided to allow the theory to run even if
        # alpha or lambda_scale are not explicitly passed.
        # Setting alpha=0 recovers the Schwarzschild metric.
        alpha = kwargs.get('alpha', 0.0) 
        lambda_scale = kwargs.get('lambda_scale', 1.0) # Must be positive and non-zero for the exponential term

        # Ensure r is a tensor and handle potential division by zero for rs/r
        r_tensor = torch.as_tensor(r, dtype=torch.float64) if not torch.is_tensor(r) else r
        
        # Numerically stable r to prevent division by zero if r approaches exactly zero
        # (Though Schwarzschild metric itself is singular at r=0)
        r_safe = torch.where(r_tensor == 0, torch.tensor(1e-9, dtype=r_tensor.dtype, device=r_tensor.device), r_tensor)

        # Schwarzschild radius calculation
        rs = 2 * G_param * M_param / C_param**2
        
        # Modified 'f' function incorporating the exponential correction
        f = 1 - rs/r_safe + alpha * torch.exp(-r_safe/lambda_scale)
        
        # Metric components for the static, spherically symmetric spacetime
        g_tt = -f
        g_rr = 1/f
        g_pp = r_safe**2  # This term corresponds to g_theta_theta and factors into g_phi_phi (r^2 sin^2(theta))
        g_tp = torch.zeros_like(r_safe) # No cross-term for static metric
        
        return g_tt, g_rr, g_pp, g_tp