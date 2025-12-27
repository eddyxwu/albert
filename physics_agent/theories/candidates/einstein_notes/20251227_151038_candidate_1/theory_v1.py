import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class ModifiedGravityLowAcceleration(GravitationalTheory):
    category = "modified_gravity"

    def __init__(self, beta=1e-10):
        super().__init__(name="Modified Gravity Low Acceleration")
        self.beta = beta
        
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        
        # Correction term that strengthens gravity at low accelerations (large r)
        # The acceleration is proportional to 1/r^2 for Schwarzschild.
        # We want a term that becomes significant as r increases.
        # A simple way is to add a term that grows with r, or decays slower than 1/r.
        # Let's try a term that is proportional to 1/r, but with a coefficient
        # that depends on the acceleration scale.
        
        # The "effective" gravitational acceleration at radius r in Schwarzschild is proportional to M/r^2.
        # We want to modify the metric such that the effective force is stronger at low accelerations.
        # A simple modification to g_tt can achieve this.
        
        # Let's consider a modification to the standard Schwarzschild metric:
        # g_tt = -(1 - rs/r + correction_term)
        # g_rr = 1 / (1 - rs/r + correction_term)
        
        # The acceleration felt by a test particle is roughly proportional to the derivative of the gravitational potential,
        # which is related to the time-time component of the metric.
        # For a static, spherically symmetric metric, the geodesic equation for a radial path gives
        # d^2r/dtau^2 = - (1/2) g_rr_inv * g_tt_prime / g_rr  * (dr/dtau)^2 / g_rr
        # The effective gravitational force is proportional to g_tt_prime / g_rr
        # For Schwarzschild, g_tt = -(1-rs/r), g_rr = 1/(1-rs/r).
        # g_tt_prime = -rs/r^2.
        # Effective force ~ (-rs/r^2) / (1/(1-rs/r)) = -rs/r^2 * (1-rs/r) ~ -rs/r^2 for large r.
        
        # We want to increase this force at large r.
        # Let's introduce a correction term that becomes significant at large r.
        # A term that is proportional to r or log(r) could work.
        # Let's try a modification to g_tt that adds a term that grows with r.
        # A simple form could be proportional to beta * r.
        # However, the units must be consistent. The correction term should be dimensionless.
        # So, let's use beta * (r / rs) or beta * (r/r0)^n for some characteristic length r0.
        # A term like beta * (r/r_char) where r_char is a large length scale,
        # or beta * (rs/r) * (r/r_char)^k where k > 1.
        
        # Let's try a modification to g_tt of the form:
        # g_tt = -(1 - rs/r + beta * (r/rs))
        # This would make the effective potential steeper at large r.
        # However, this term grows with r, which might cause issues.
        
        # Alternative: modify the effective potential in a way that it doesn't go to zero as fast as 1/r.
        # Let's try adding a term to the Ricci scalar in an f(R) context, but directly modifying metric components.
        
        # Let's try a correction to g_tt that is proportional to rs/r and decays slower than 1/r.
        # Consider a term like beta * (rs/r)^p where p < 1.
        # For example, p = 1/2.
        # g_tt = -(1 - rs/r + beta * sqrt(rs/r))
        # The acceleration derivative would be approximately -rs/r^2 - beta * (1/2) * sqrt(rs) * r^(-3/2).
        # This adds a term that decays slower than 1/r^2.
        
        # Let's try a simpler approach by directly modifying g_tt with a term that's significant at large r.
        # We want gravity to be stronger at low accelerations.
        # This means the effective potential should be deeper or its gradient steeper at large distances.
        # The standard potential is Phi ~ -GM/r. The force is F ~ -dPhi/dr ~ -GM/r^2.
        # We want F to be larger (more attractive) than -GM/r^2 at large r.
        
        # Let's introduce a correction term to g_tt that adds to the Schwarzschild potential.
        # The correction should be small at small r and grow or decay slower than 1/r at large r.
        
        # A common approach in MOND-like theories is to have a crossover acceleration a0.
        # At accelerations a >> a0, Newtonian gravity holds.
        # At accelerations a << a0, acceleration is proportional to force (a ~ F).
        
        # Let's try a correction to g_tt of the form:
        # g_tt = -(1 - rs/r + beta * r)
        # This is dimensionally inconsistent without scaling.
        
        # Let's scale beta by rs:
        # g_tt = -(1 - rs/r + beta * rs * (r/rs)) = -(1 - rs/r + beta * r)
        # Still dimensionally problematic.
        
        # Let's consider a dimensionless correction.
        # We want the effective gravitational potential term to be more negative at large r.
        # Standard: -(1 - rs/r)
        # Modified: -(1 - rs/r + f_corr(r)) where f_corr(r) is positive and grows with r or decays slower than 1/r.
        
        # Let's try a correction proportional to the inverse of the acceleration scale.
        # The acceleration scale is a0. Let's assume beta is related to 1/a0.
        # The acceleration is proportional to M/r^2.
        # We want a modification when M/r^2 < a0.
        
        # Let's try a correction term to g_tt that is proportional to rs/r but with a modified exponent or a sum.
        # Consider:
        # g_tt = -(1 - rs/r + beta * (rs/r)**p)
        # If p < 1, the correction term grows as r increases.
        # Let's choose p = 0.5 for simplicity (square root).
        
        correction_term = self.beta * torch.sqrt(rs / r)
        
        # Ensure the correction doesn't make the metric pathological at small r
        # The term rs/r dominates at small r. We need 1 - rs/r + correction_term > 0 for g_rr to be positive.
        # Or rather, we need 1 - rs/r + correction_term > 0 for the denominator of g_rr.
        # The standard Schwarzschild has a singularity at r=rs.
        # Our correction term sqrt(rs/r) decreases as r increases.
        # At r=rs, correction_term = beta.
        # g_tt = -(1 - 1 + beta * sqrt(1)) = -beta
        # g_rr = 1 / (-beta) = -1/beta. This is problematic.
        
        # Let's rethink the correction term.
        # We want gravity to strengthen at low accelerations.
        # This means the effective potential should be deeper.
        # The potential is related to g_tt.
        # Let's try to ADD to the magnitude of g_tt.
        # g_tt = -(1 - rs/r) - correction
        # where 'correction' is positive and becomes significant at large r.
        
        # A simple correction could be beta * (rs/r)^p with p < 1.
        # Let's use p = 0.5.
        # correction = beta * sqrt(rs/r)
        
        # g_tt = -(1 - rs/r + beta * sqrt(rs/r))
        # g_rr = 1 / (1 - rs/r + beta * sqrt(rs/r))
        
        # Let's analyze the behavior at large r:
        # g_tt approx -(1 + beta * sqrt(rs/r))
        # g_rr approx 1 / (1 + beta * sqrt(rs/r)) approx 1 - beta * sqrt(rs/r)
        
        # The effective potential Phi is related to g_tt = -(1 + 2*Phi/c^2).
        # So, Phi approx -(rs/r + beta * sqrt(rs/r)) * C_param**2 / 2.
        # The force F = -dPhi/dr.
        # dPhi/dr approx -( -rs/r^2 - beta * (1/2) * sqrt(rs) * r**(-3/2) ) * C_param**2 / 2
        # F approx (rs/r^2 + (beta/2) * sqrt(rs) * r**(-3/2)) * C_param**2 / 2
        
        # The first term is the Newtonian force.
        # The second term is the correction, which decays slower than 1/r^2.
        # This means the force is stronger than Newtonian at large r.
        
        # Let's ensure that the denominator of g_rr remains positive.
        # 1 - rs/r + beta * sqrt(rs/r) > 0
        # Let x = sqrt(rs/r). Then 1 - x^2 + beta*x > 0.
        # This is a quadratic in x. The roots are ( -beta +/- sqrt(beta^2 + 4) ) / 2.
        # Since x = sqrt(rs/r) is always positive, we need the positive root.
        # x_root = (-beta + sqrt(beta^2 + 4)) / 2.
        # As beta > 0, x_root is always positive.
        # The quadratic 1 - x^2 + beta*x is a downward parabola.
        # It is positive between its roots.
        # The roots are approximately (-beta +/- 2)/2.
        # The positive root is approximately (2 - beta)/2 = 1 - beta/2.
        # The negative root is approximately (-2 - beta)/2 = -1 - beta/2.
        # So, the condition is -1 - beta/2 < x < 1 - beta/2.
        # Since x = sqrt(rs/r) > 0, we need 0 < x < 1 - beta/2.
        # This means sqrt(rs/r) < 1 - beta/2.
        # rs/r < (1 - beta/2)^2.
        # r > rs / (1 - beta/2)^2.
        # This means the metric is well-behaved for r larger than some value.
        # This is acceptable as it's a modification at large scales.
        
        f = 1 - rs/r + self.beta * torch.sqrt(rs / r)
        
        # Ensure f is positive to avoid issues with g_rr.
        # We can clip f or ensure beta is small enough.
        # For simplicity, let's assume beta is small enough for typical r.
        # Or we can clip f to a small positive value if it goes negative.
        
        # Let's ensure f > epsilon
        epsilon = 1e-10
        f = torch.maximum(f, torch.tensor(epsilon, device=r.device))
        
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        return g_tt, g_rr, g_pp, g_tp