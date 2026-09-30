import numpy as np
from einops import einsum


def decay_spectrum(variance, rate, batch, decay):
    """Positive quartic roots for noise variances normalized to mean one.

    Solve (gamma + strength)*s^4 - gamma*s^3 + gamma*s
    - (gamma + strength*variance) = 0 for each singular value s.
    Results retain the input order of the variances.
    """
    strength = rate * len(variance) / batch
    gamma = decay * (1 + rate*decay)
    result = []
    for value in variance:
        roots = np.roots([gamma+strength, -gamma, 0, gamma, -(gamma+strength*value)])
        positive = roots[np.isreal(roots) & (roots.real > 0)].real
        if len(positive) != 1:
            raise ValueError("Expected exactly one positive quartic root")
        result.append(positive[0])
    return np.array(result)


def chain_scales(variance, depth):
    """Interior scale c and boundary scale a: singular values c, a*lambda^(±1/4)."""
    mean_root = np.sqrt(variance).mean()
    interior_scale = mean_root**(1/depth)
    boundary_scale = mean_root**(-(depth-2)/(2*depth))
    return interior_scale, boundary_scale


def symmetric_power(matrix, power):
    """Raise a symmetric positive-definite matrix to a power through its eigenvalues."""
    values, vectors = np.linalg.eigh(matrix)
    return einsum(vectors, values**power, vectors, "row mode, mode, col mode -> row col")


def predicted_gram(input_covariance, noise_covariance):
    """Unique minimizer of the on-manifold entropic objective for positive-definite covariances."""
    root = symmetric_power(input_covariance, .5)
    inverse = np.linalg.inv(root)
    ratio = np.trace(input_covariance) / np.trace(noise_covariance)
    scaled_covariance = ratio * einsum(root, noise_covariance, root,
                                       "row inner, inner middle, middle col -> row col")
    balanced = symmetric_power(scaled_covariance, 0.5)
    return einsum(inverse, balanced, inverse, "row inner, inner middle, middle col -> row col")
