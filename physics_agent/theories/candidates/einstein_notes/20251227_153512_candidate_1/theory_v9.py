import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Oscillatory Modified Schwarzschild Metric.

    This theory proposes a modification to the standard Schwarzschild metric by introducing
    an oscillatory correction term that depends on the radial coordinate `r` and a characteristic
    scale `lambda_scale`. Such oscillatory corrections can arise in various extensions of General
    Relativity, particularly in scenarios involving:
    1.  **Effective Field Theories:** As an effective description of quantum gravitational effects
        or unknown physics at a specific energy scale, where `lambda_scale` represents this characteristic length scale.
    2.  **Scalar-Tensor Theories:** If gravity couples to a scalar field whose potential or dynamics
        exhibit oscillatory behavior, these oscillations can be imprinted on the spacetime metric.
    3.  **Non-local Gravity:** Theories of non-local gravity, where the gravitational action involves
        integrals over spacetime, can naturally lead to non-trivial functional dependencies, including
        oscillations, in the metric solutions.
    4.  **Modified Matter/Dark Energy Interactions:** Specific interactions between standard matter/dark energy
        and a modified gravitational sector could also induce such periodic deviations.

    A potential Lagrangian formulation that could lead to such a metric modification
    might involve an action with a non-local term or a scalar field with an oscillatory potential:

    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} (R + \mathcal{L}_{non-local}) + L_{matter} \right]
    or
    S = \int d^4x \sqrt{-g} \left[ \frac{1}{16\pi G} R - \frac{1}{2} g^{\mu\nu} \partial_\mu \phi \partial_\nu \phi - V(\phi) + L_{matter} \right]

    where in the first case, \mathcal{L}_{non-local} represents a non-local modification to the Ricci scalar,
    and in the second case, V(\phi) is an oscillatory potential for the scalar field \phi (e.g., V(\phi) ~ cos(\phi/f)).
    The oscillatory correction term in the metric is a phenomenological representation of these effects.

    The parameter \alpha controls the amplitude of the oscillatory modification,
    and \lambda_scale sets its characteristic wavelength. This metric represents a static,
    spherically symmetric vacuum solution in such a modified gravity theory.
    """
    category = "modified_gravity"
    
    def __init__(self, alpha=0.01, lambda_scale=1e5): # lambda_scale in meters, e.g., 100 km
        super().__init__(name="OscillatoryCustomTheory", force_6dof_solver=False)
        self.alpha = alpha  # Amplitude of the oscillatory correction
        self.lambda_scale = lambda_scale # Characteristic wavelength of the oscillation
    
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # Ensure r is positive and not zero to avoid division by zero
        r_safe = torch.maximum(r, torch.tensor(1e-10, device=r.device))
        
        # Oscillatory correction term: alpha * sin(r/lambda_scale) / r^2
        # This is a different mathematical form as requested, aiming for a new approach.
        
        oscillatory_correction = self.alpha * torch.sin(r_safe / self.lambda_scale) / (r_safe**2)
        
        rs_over_r = rs / r_safe
        
        # The 'f' component of the metric, analogous to (1 - rs/r) in Schwarzschild
        f = 1 - rs_over_r + oscillatory_correction
        
        # Ensure 'f' does not become zero or negative to avoid division by zero or imaginary metrics
        # This clamp is crucial for numerical stability, especially near potential horizons or singularities
        f = torch.maximum(f, torch.tensor(1e-10, device=r.device))
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r_safe**2 # Represents the angular component g_theta_theta * sin^2(theta) or g_phi_phi. For Schwarzschild, g_theta_theta = r^2.
        g_tp = torch.zeros_like(r) # For a static, non-rotating metric, g_tp (or g_tphi) is zero
        
        return g_tt, g_rr, g_pp, g_tp