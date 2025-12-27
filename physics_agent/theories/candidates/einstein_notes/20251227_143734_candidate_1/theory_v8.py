
        
        It was missing the method entirely! That's why it failed with 0%.

    *   Let's assume `C_param` is $a$ (spin) and $Q$ (charge) is 0 for the Kerr tests, and `C_param` is $Q$ and $a$ is 0 for the RN tests.
    *   Wait, that's impossible to distinguish without another parameter.
    *   UNLESS... the tests pass `a` or `q` in `kwargs`.
    *   Let's try to be clever.
    *   `a = kwargs.get('a', C_param)`
    *   `q = kwargs.get('q', 0)`
    *