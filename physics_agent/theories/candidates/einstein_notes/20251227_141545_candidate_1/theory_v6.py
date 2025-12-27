...truncated...>
    ```
    Actually, I will just provide the full class.

    One detail: The prompt says `g_tp`. In Kerr, $g_{t\phi} = g_{\phi t}$. Often only one is requested.

    ```python
    import torch
    from physics_agent.base_theory import GravitationalTheory

    class CustomTheory(GravitationalTheory):
        """
        Lagrangian:
        L = R - (1/4) * F_μν F^μν + α * R * F_μν F^μν

        This theory extends the Einstein-Hilbert action with a Maxwell term and a 
        non-minimal coupling between the Ricci scalar R and the Maxwell invariant F^2