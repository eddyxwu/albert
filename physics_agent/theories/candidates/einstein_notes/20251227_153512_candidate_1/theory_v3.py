import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Power-Law Modified Schwarzschild Metric.

    This theory proposes a modification to the standard Schwarzschild metric by introducing an
    inverse power-law correction term. Such corrections can arise from various extensions
    of General Relativity, particularly theories involving higher-order curvature invariants
    in the gravitational action. For instance, terms like R^2 or R_mu_nu R^mu_nu in the
    Lagrangian can lead to modified field equations whose solutions, in certain limits,
    exhibit inverse power-law deviations from the Schwarzschild metric in static,
    spherically symmetric vacuum spacetimes.

    A potential Lagrangian formulation that could lead to such a metric modification
    is an action of the form:

    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} (R + f(R, R_{\mu\nu}R^{\mu\nu}, \dots)) + L_{matter} \right]

    where f is a function of curvature invariants that includes higher-order terms. For example,
    a simple f(R) gravity model, where R is the Ricci scalar, could be R + \beta R^2.
    Alternatively, theories with extra dimensions (e.g., braneworld models) can also induce
    inverse power-law corrections to the gravitational potential in the observable 4D spacetime.

    The parameters \alpha and n in the metric correction term (\alpha (rs/r)^n) are
    phenomenological coefficients related to the coupling constants of these higher-order
    terms or the specific geometry of extra dimensions. This metric represents a static,
    spherically symmetric vacuum solution in such a modified gravity theory.
    """
    category = "modified_gravity"
    
    def __init__(self, alpha=0.01, n=3.0): 
        super().__init__(name="PowerLawCustomTheory", force_6dof_solver=False)
        self.alpha = alpha  # Strength of the power-law correction
        self.n = n          # Exponent of the power-law correction (e.g., 2, 3, or 0.5)
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # Power-law correction term: alpha * (rs/r)**n
        correction = self.alpha * (rs / r)**self.n
        
        f = 1 - rs/r + correction
        
        # Ensure 'f' does not become zero or negative to avoid division by zero or imaginary metrics
        f = torch.maximum(f, torch.tensor(1e-10, device=r.device))
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r**2 # Represents the angular component g_theta_theta * sin^2(theta) or g_phi_phi
        g_tp = torch.zeros_like(r) # For a static, non-rotating metric
        
        return g_tt, g_rr, g_pp, g_tp