"""
Example potential functions.
"""

import numpy as np


def free_particle_potential() -> 'nm -> kJ/mol':
    """
    Free particle potential.
    """

    def free_particle(q: 'nm') -> 'kJ/mol':
        # Remove the inner-most dimension.
        return np.zeros(q.shape[:-1])

    return free_particle


def harmonic_potential(k: 'kJ/mol nm^2' = None, m: 'g/mol' = None, w: '1/ps' = None) -> 'nm -> kJ/mol':
    """
    Harmonic potential relative to the origin.

    Note:
      Either k or (m and w) must be specified.

    Parameters:
      k: Spring constant.
      m: Mass of particle.
      w: Angular frequency of oscillator.
    """

    if k is not None:
        force_constant = k  # kJ/mol nm^2
    elif m is not None and w is not None:
        force_constant = m * w * w  # kJ/mol nm^2
    else:
        assert False, 'Must provide either k or (m and w).'

    def harmonic(q: 'nm') -> 'kJ/mol':
        return force_constant * q[..., 0] * q[..., 0] / 2

    return harmonic
