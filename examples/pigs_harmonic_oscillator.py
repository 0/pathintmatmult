#!/usr/bin/env python3

"""
Harmonic oscillator PIGS example.

An oscillator with an angular frequency of x kelvin has a ground state
potential energy of x/4 kelvin and a total energy of x/2 kelvin. One with a
mass of 1 electron mass and angular frequency of 1 K has a spread of about 120
nm in either direction from the origin; one with a mass of 10 electron masses
spreads about 40 nm. The following are some possible combinations of arguments
to try:

  --mass 1 --omega 1 --grid-range 120 --grid-len 100 --beta 12 --num-links 1200

  --mass 10 --omega 1 --grid-range 40 --grid-len 100 --beta 12 --num-links 1200

If --trial-deform is not given, a uniform trial function is used. If it is
given, the exact ground state is used as the trial fuction, but is deformed by
the given factor (1 corresponds to no deformation).
"""

from argparse import ArgumentParser

import numpy as np

from pathintmatmult import PIGSMM
from pathintmatmult.constants import HBAR, KB, ME
from pathintmatmult.potentials import harmonic_potential


# Parse arguments.
p = ArgumentParser(description='Calculate HO ground state properties using PIGSMM.')
p_config = p.add_argument_group('configuration')

p_config.add_argument('--mass', metavar='M', type=float, required=True, help='particle mass (electron masses)')
p_config.add_argument('--omega', metavar='W', type=float, required=True, help='angular frequency (K)')
p_config.add_argument('--grid-range', metavar='R', type=float, required=True, help='grid range from origin (nm)')
p_config.add_argument('--grid-len', metavar='L', type=int, required=True, help='number of points on grid')
p_config.add_argument('--beta', metavar='B', type=float, required=True, help='propagation length (1/K)')
p_config.add_argument('--num-links', metavar='P', type=int, required=True, help='number of links')
p_config.add_argument('--trial-deform', metavar='D', type=float, help='deformation factor for exact trial function')

p.add_argument('--wf-out', metavar='FILE', help='path to output wavefunction values')
p.add_argument('--density-out', metavar='FILE', help='path to output density plot')

args = p.parse_args()

mass = args.mass * ME  # g/mol
omega = args.omega * KB / HBAR  # 1/ps
grid_range = args.grid_range  # nm
grid_len = args.grid_len  # 1
beta = args.beta / KB  # mol/kJ
num_links = args.num_links  # 1
trial_deform = args.trial_deform

wf_out = args.wf_out
density_out = args.density_out


# Calculate values.
harmonic = harmonic_potential(m=mass, w=omega)
kwargs = {}

if trial_deform is not None:
    alpha = trial_deform * mass * omega / HBAR  # 1/nm^2

    def trial_f(q: 'nm') -> '1':
        return np.exp(-0.5 * alpha * q[..., 0] ** 2)

    def trial_f_diff(q: 'nm') -> '1/nm^2':
        return alpha * (alpha * q[..., 0] ** 2 - 1) * trial_f(q)

    kwargs['trial_f'] = trial_f
    kwargs['trial_f_diffs'] = [trial_f_diff]

ho_pigs = PIGSMM([mass], [grid_range], [grid_len], beta, num_links, harmonic, **kwargs)

estimated_potential_energy = ho_pigs.expectation_value(harmonic) / KB  # K
estimated_total_energy = ho_pigs.energy_mixed / KB  # K

print('V = {} K'.format(estimated_potential_energy))
# According to the virial theorem, <K> = <V> for a harmonic oscillator.
print('E_virial = {} K'.format(2 * estimated_potential_energy))
print('E_mixed = {} K'.format(estimated_total_energy))


# Output wavefunction.
if wf_out:
    np.savetxt(wf_out, np.dstack((ho_pigs.grid, ho_pigs.ground_wf))[0])

# Output plot.
if density_out:
    from pathintmatmult.plotting import plot2d

    xy_range = (-grid_range, grid_range)

    plot2d(ho_pigs.density, xy_range, xy_range, density_out, x_label=r'$q_j / \mathrm{nm}$', y_label=r'$q_i / \mathrm{nm}$')
