import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    category = "modified_gravity"

    def __init__(self, beta=1e-10, alpha=1e-10):
        super().__init__(name="Custom Modified Gravity")
        self.beta = beta
        self.alpha = alpha
        
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # This theory aims to modify gravity at low accelerations, inspired by MOND.
        # The modification is introduced by altering the time-time component of the metric, g_tt.
        # The standard Schwarzschild metric is g_tt = -(1 - rs/r).
        # We introduce a correction term that becomes significant at large r (low accelerations).
        #
        # The proposed modification to g_tt is:
        # g_tt = -(1 - rs/r + beta * (rs/r)^p - alpha * (rs/r)^q)
        # where beta and alpha are coupling constants, and p, q are exponents.
        #
        # To strengthen gravity at low accelerations (large r), we need the magnitude of g_tt to be larger,
        # or equivalently, the effective potential to be deeper.
        #
        # Let's consider the effective potential Phi, where g_tt = -(1 + 2*Phi/c^2).
        # In Schwarzschild, Phi_schw = -GM/r.
        #
        # We want to add a term to the potential that decays slower than 1/r.
        # A term like (rs/r)^p with p < 1 would achieve this.
        #
        # Let's use p = 0.5, so we add beta * sqrt(rs/r). This strengthens gravity at large r.
        #
        # However, the g-2 Muon test failure suggests a deviation from standard behavior.
        # The original code had `beta * torch.sqrt(rs / r)`.
        # Let's try to incorporate a term that might affect higher-order derivatives or introduce a different
        # functional form that can be tuned.
        #
        # A common approach for MOND-like theories is to introduce a crossover acceleration a0.
        # The transition from Newtonian to MONDian behavior occurs around this acceleration.
        #
        # Let's try a modification that explicitly depends on the ratio of acceleration to a characteristic acceleration.
        # The acceleration in Schwarzschild is proportional to M/r^2.
        # Let's relate beta to 1/a0.
        #
        # Let's try a functional form that is a combination of terms decaying at different rates.
        # Consider a correction term of the form: beta * (rs/r)^p - alpha * (rs/r)^q.
        # If p < 1 and q > 1, this term can be positive and significant at large r, then decay.
        #
        # For the g-2 Muon anomaly, a possible explanation involves modifications to the photon or
        # graviton self-energy. In metric theories, this often relates to higher-order curvature invariants
        # or non-minimal couplings.
        #
        # Let's consider a modification inspired by some scalar-tensor theories or f(R) gravity,
        # but implemented directly in the metric.
        #
        # The current formulation `f = 1 - rs/r + self.beta * torch.sqrt(rs / r)` leads to
        # `g_tt = -f` and `g_rr = 1/f`.
        #
        # The g-2 Muon anomaly suggests a deviation from the expected magnetic moment.
        # This could arise from modifications to the electromagnetic interaction mediated by gravity.
        #
        # Let's try a correction that is proportional to the curvature.
        # For Schwarzschild, the Ricci scalar R = 0. The Kretschmann scalar K = 48*rs^2/r^6.
        #
        # A simple modification to g_tt that strengthens gravity at low acceleration (large r)
        # and might address the g-2 anomaly could involve terms that grow with distance or decay slower than 1/r.
        #
        # Let's adjust the exponent of the correction term.
        # The previous form was beta * (rs/r)^0.5.
        #
        # Let's try a form that has a crossover behavior.
        # Consider a correction of the form: beta * (rs/r) / (1 + alpha * (rs/r)^k)
        # Or a simpler form: beta * (rs/r)^p.
        #
        # The failure in "Quantum Geodesic Sim" might indicate issues with causality or
        # pathological behavior at certain radii or for certain parameter values.
        #
        # Let's try to refine the existing `beta * sqrt(rs/r)` term, possibly by adjusting its exponent or adding another term.
        #
        # A common modification that addresses low acceleration phenomena is adding a term that grows with r.
        # However, for consistency, it should be dimensionless.
        #
        # Let's try to generalize the exponent:
        # `correction = beta * (rs/r)**exponent`
        # If exponent < 1, gravity is stronger at large r.
        # If exponent = 0.5, we get sqrt(rs/r).
        #
        # For the g-2 muon anomaly, it's often related to new particles or interactions.
        # In a purely metric theory, this might manifest as a deviation from GR's prediction for
        # the effective gravitational potential experienced by charged particles.
        #
        # Let's try a correction that involves a power law, but with a tuneable exponent.
        #
        # The previous `f = 1 - rs/r + self.beta * torch.sqrt(rs / r)` might be too simple.
        # Let's try a form that can interpolate between Newtonian and a modified regime.
        #
        # Consider a modification to the `rs/r` term itself.
        # `rs_eff = rs * (1 + beta * (r/r0)**n)`
        # This would make the effective mass grow with distance.
        #
        # Let's stick to modifying `g_tt` directly.
        #
        # The original code: `f = 1 - rs/r + self.beta * torch.sqrt(rs / r)`
        #
        # Let's try to adjust the exponent of the correction term.
        # If `beta * sqrt(rs/r)` is causing issues, maybe a different power or a sum of powers is needed.
        #
        # The "Quantum Geodesic Sim" failure might be due to the metric becoming singular or
        # non-causal at certain radii.
        # `1 - rs/r + beta * sqrt(rs/r)` must remain positive for `g_rr`.
        # As analyzed before, this requires `r > rs / (1 - beta/2)^2`.
        #
        # Let's try to introduce a new parameter that controls the "strength" of the modification
        # at different scales, potentially addressing the g-2 anomaly.
        #
        # A simple modification to `g_tt` that could affect particle properties is to add a term
        # that depends on the radial coordinate in a non-trivial way.
        #
        # Let's try a correction of the form `beta * (rs/r)**p` where `p` is tunable.
        # The original used `p=0.5`.
        #
        # For the g-2 Muon anomaly, some theories propose modifications to the photon propagator or
        # vacuum polarization effects due to exotic fields. In a metric theory context, this could
        # be indirectly modeled by how the metric affects charged particle dynamics.
        #
        # Let's try a modification that has a more pronounced effect at larger radii,
        # and potentially a different behavior at smaller radii compared to the simple sqrt term.
        #
        # Consider a modification of the form:
        # `g_tt = -(1 - rs/r + beta * (rs/r)**p - alpha * (rs/r)**q)`
        # with `p < 1` and `q > 1`.
        #
        # Let's try `p = 0.5` and `q = 2.0`.
        # The term `- alpha * (rs/r)**2` would decay faster than `rs/r` and `beta * sqrt(rs/r)` at large r.
        # This might help in passing the Quantum Geodesic Sim by ensuring the metric remains well-behaved.
        #
        # Let's redefine `f` as:
        # `f = 1 - rs/r + beta * (rs/r)**0.5 - alpha * (rs/r)**2`
        #
        # The choice of exponents is crucial. `p=0.5` provides the low-acceleration enhancement.
        # `q=2.0` provides a term that decays faster than `rs/r` at large `r`, potentially
        # helping to regularize the metric or avoid unwanted behavior.
        #
        # The `beta` parameter controls the strength of the low-acceleration enhancement.
        # The `alpha` parameter controls the strength of a higher-order term.
        #
        # Let's analyze the condition `f > epsilon`.
        # `1 - rs/r + beta * sqrt(rs/r) - alpha * (rs/r)**2 > epsilon`
        # Let `x = sqrt(rs/r)`. Then `1 - x^2 + beta*x - alpha*x^4 > epsilon`.
        # This is a quartic inequality. For small `x` (large `r`), the dominant terms are `1 + beta*x`.
        # For large `x` (small `r`), the `-alpha*x^4` term dominates and makes `f` negative if `alpha` is large.
        # This implies that the theory might have a singularity at smaller radii, which is expected
        # as we are modifying gravity.
        #
        # The g-2 Muon anomaly is a subtle effect and might require specific parameter tuning.
        # The `beta` parameter primarily influences the low-acceleration regime.
        # The `alpha` parameter influences behavior at intermediate and small radii.
        #
        # Let's set default values for `beta` and `alpha` that are small, assuming they are
        # deviations from GR.
        
        # Effective Schwarzschild radius
        rs = 2 * G_param * M_param / C_param**2
        
        # Dimensionless radial coordinate scaled by rs
        r_scaled = r / rs
        
        # Correction term inspired by MOND-like theories (enhances gravity at low acceleration)
        # Use beta * (rs/r)^0.5
        low_accel_correction = self.beta * torch.pow(rs / r, 0.5)
        
        # Additional term to potentially address g-2 anomaly and improve metric behavior.
        # Using a term that decays faster at large r, e.g., alpha * (rs/r)^2.
        # This term might help regularize the metric at intermediate radii.
        high_order_correction = self.alpha * torch.pow(rs / r, 2.0)
        
        # Combine terms for the f(r) function in the metric
        # f = 1 - rs/r + low_accel_correction - high_order_correction
        f_val = 1.0 - (rs / r) + low_accel_correction - high_order_correction
        
        # Ensure f_val remains positive to avoid issues with g_rr (event horizon).
        # The theory is expected to deviate from Schwarzschild at large radii,
        # and potentially have different horizon structures or singularities.
        # We clip f_val to a small positive epsilon to prevent division by zero or negative values.
        epsilon = 1e-15
        f_val = torch.maximum(f_val, torch.tensor(epsilon, device=r.device))
        
        g_tt = -f_val
        g_rr = 1.0 / f_val
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp