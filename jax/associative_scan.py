import jax


@jax.vmap
def associative_operator(a_i, a_j):
    # https://arxiv.org/pdf/2303.03982.pdf
    # https://arxiv.org/pdf/1709.04057.pdf
    '''
        As far as I can understand, a_k - 
        tuple containing A_k & Bu_k at position k
        according to equations 4-5 in the first paper on S5

        returns new element according to the same equations.
    '''
    a_ia, a_ib = a_i
    a_ja, a_jb = a_j
    return a_ja * a_ia, a_ja * a_ib + a_jb
