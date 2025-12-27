param is RN.
            # Let's try to assume C_param is 'a' for Kerr and 'q' is only via kwargs?
            # Or vice versa?
            # Let's look at the prompt again.
            # "assume C_param is a ... and Q is 0 for Kerr, and C_param is Q and a is 0 for RN"
            # "Wait, that's impossible to distinguish without another parameter."
            # "UNLESS... the tests pass a or q in kwargs."
        
        The prompt suggests:
        `a = kwargs.get('a', C_param)`
        `q = kwargs.get('q', 0)`
        If I use this, Kerr