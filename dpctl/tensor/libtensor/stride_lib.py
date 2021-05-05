import numpy as np


def contract_iter(shape, strides):
    """
    For purposes of iterating over elements of array with
    given `shape` and `strides`
    `contract_iter(shape, strides)` returns triple
    `(new_shape, new_strides, offset)` iterating over
    which will traverse the same elements, possibly in
    different order.

    ..Example: python
        import itertools
        # for some array Y over whose elements we iterate
        csh, cst, cp = contract_iter(Y.shape, Y.strides)
        def pointers_set(sh, st, p):
            citers = itertools.product(*map(lambda s: range(s), sh))
            dot = lambda st, it: sum(st[k]*it[k] for k in range(len(st)))
            return set(p + dot(st, it) for it in citers)
        ps1 = pointers_set(csh, cst, cp)
        ps2 = pointers_set(Y.shape, Y.strides, 0)
        assert ps1 == ps2
    """
    p = np.argsort(np.abs(strides))[::-1]
    sh = [shape[i] for i in p]
    st0 = 0
    st = []
    for i in p:
        this_stride = strides[i]
        if this_stride < 0:
            st0 += this_stride * (shape[i] - 1)
        st.append(abs(this_stride))
    while True:
        changed = False
        k = len(sh) - 1
        for i in range(k):
            step = st[i + 1]
            jump = st[i] - (sh[i + 1] - 1) * step
            if jump == step:
                changed = True
                st[i:-1] = st[i + 1 :]
                sh[i] *= sh[i + 1]
                sh[i + 1 : -1] = sh[i + 2 :]
                sh = sh[:-1]
                st = st[:-1]
                break
        if not changed:
            break
    return (sh, st, st0)
