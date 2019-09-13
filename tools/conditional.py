from theta import rtbm
import numpy as np





def conditional(model, d):
    """Generates the conditional RTBM. 
    
    Args:
        model (theta.rtbm.RTBM): RTBM modelling the original PDF
        d (numpy.array): column vector containing the values for the conditional

    Returns:
        theta.rtbm.RTBM: RTBM modelling the conditional probability P(y|d)
    """

    nh = model._Nh
    nv = model._Nv 
    
    assert (nv > 1), "cannot do the conditional probability of a 1d distribution"

    assert (d.size < nv), "d larger than Nv"

    k = int(nv-d.size)
    
    cmodel = rtbm.RTBM(k, nh)

    # Matrix A
    t = model.t[:k,:k]
    t = t[np.triu_indices(k)]
    q = model.q[np.triu_indices(nh)]
    w = model.w[:k]

    # Biases
    bh = model.bh + np.dot(model.w[k:].T, d)
    bv = model.bv[:k] + np.dot(model.t[:k,k:], d)

    cparams = np.concatenate((bv, bh, w, t, q), axis = None)
    cmodel.set_parameters(cparams)

    return cmodel
