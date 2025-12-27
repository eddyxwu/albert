import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Lagrangian:
    L = R - (1/4) * F_μν F^μν + α * R * F_μν F^μν

    This theory extends the Einstein-Hilbert action with a Maxwell term and a 
    non-minimal coupling between the Ricci scalar R and the Maxwell invariant F^2. 
    The term α * R * F^2 introduces a coupling between the curvature and the 
    electromagnetic field, leading to corrections in the metric that scale with 
    higher powers of 1/r. In the weak field limit, this modifies the 
    Reissner-Nordström solution by introducing an effective r-dependent charge 
    term, which recovers the standard GR solutions (Schwarzschild, Kerr, 
    Reissner-Nordström) when α → 0 or Q → 0.
    """

    def get_metric(self, r, M, a, Q, alpha):
        """
        Returns the metric components for the modified Kerr-Newman-like solution.
        
        Args:
            r: Radial coordinate (torch.Tensor)
            theta: Polar angle (torch.Tensor)
            M: Mass parameter
            a: Angular momentum parameter (spin)
            Q: Charge parameter
            alpha: Non-minimal coupling constant
            
        Returns:
            tuple: (g_tt, g_rr, g_pp, g_tp)
        """
        # Define auxiliary functions for the Kerr-like geometry
        sigma = r**2 + a**2 * torch.cos(theta)**2
        
        # Effective potential term including the non-minimal coupling correction
        # The coupling alpha * R * F^2 modifies the effective charge contribution
        # to the metric function Delta.
        delta = r**2 - 2*M*r + a**2 + Q**2 * (1 - 2 * alpha / r**2)
        
        # Metric components in Boyer-Lindquist-like coordinates
        # g_tt: Time-time component
        g_tt = -(1 - (2 * M * r - Q**2 * (1 - 2 * alpha / r**2)) / sigma)
        
        # g_rr: Radial-radial component
        g_rr = sigma / delta
        
        # g_pp: Phi-phi component (angular)
        g_pp = (r**2 + a**2 + (2 * M * r - Q**2 * (1 - 2 * alpha / r**2)) * a**2 * torch.sin(theta)**2 / sigma) * torch.sin(theta)**2
        
        # g_tp: Time-phi component (off-diagonal/rotation)
        g_tp = -(2 * M * r - Q**2 * (1 - 2 * alpha / r**2)) * a * torch.sin(theta)**2 / sigma
        
        return g_tt, g_rr, g_pp, g_tp