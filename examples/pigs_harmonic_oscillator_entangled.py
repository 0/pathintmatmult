#!/usr/bin/env python3

"""
Entangled harmonic oscillators PIGS example.

A pair of identical harmonic oscillators with a harmonic interaction potential.
"""

from argparse import ArgumentParser

import numpy as np

from pathintmatmult import PIGSIMM
from pathintmatmult.constants import HBAR, KB, ME
from pathintmatmult.potentials import harmonic_potential


# Parse arguments.
p = ArgumentParser(description='Calculate entangled HO ground state properties using PIGSMM2.')
p_config = p.add_argument_group('configuration')

p_config.add_argument('--mass', metavar='M', type=float, required=True, help='particle mass (electron masses)')
p_config.add_argument('--omega-0', metavar='W', type=float, required=True, help='central potential angular frequency (K)')
p_config.add_argument('--omega-int', metavar='W', type=float, required=True, help='interaction potential angular frequency (K)')
p_config.add_argument('--grid-range', metavar='R', type=float, required=True, help='grid range from origin (nm)')
p_config.add_argument('--grid-len', metavar='L', type=int, required=True, help='number of points on grid')
p_config.add_argument('--beta', metavar='B', type=float, required=True, help='propagation length (1/K)')
p_config.add_argument('--num-links', metavar='P', type=int, required=True, help='number of links')
p_config.add_argument('--trial-deform', metavar='D', type=float, help='deformation factor for exact trial function')

p.add_argument('--wf-out', metavar='FILE', help='path to output wavefunction values')
p.add_argument('--density-diagonal-out', metavar='FILE', help='path to output diagonal density plot')

args = p.parse_args()

mass = args.mass * ME  # g/mol
omega_0 = args.omega_0 * KB / HBAR  # 1/ps
omega_int = args.omega_int * KB / HBAR  # 1/ps
grid_range = args.grid_range  # nm
grid_len = args.grid_len  # 1
beta = args.beta / KB  # mol/kJ
num_links = args.num_links  # 1
trial_deform = args.trial_deform

wf_out = args.wf_out
density_diagonal_out = args.density_diagonal_out


# Calculate values.
pot_0 = harmonic_potential(m=mass, w=omega_0)
pot_int = harmonic_potential(m=mass, w=omega_int)


def total_potential(qs: '[nm]') -> 'kJ/mol':
    return pot_0(qs[..., [0]]) + pot_0(qs[..., [1]]) + pot_int(qs[..., [0]] - qs[..., [1]])

kwargs = {}

if trial_deform is not None:
    alpha = trial_deform * mass / HBAR  # ps/nm^2
    omega_R = omega_0  # 1/ps
    omega_r = np.sqrt(omega_0 * omega_0 + 2 * omega_int * omega_int)  # 1/ps
    omega_p = omega_R + omega_r  # 1/ps
    omega_m = omega_R - omega_r  # 1/ps

    def trial_f(qs: '[nm]') -> '1':
        return np.exp(-0.25 * alpha * (omega_p * (qs[..., 0] ** 2 + qs[..., 1] ** 2) + 2 * omega_m * qs[..., 0] * qs[..., 1]))

    def trial_f_diff_0(qs: '[nm]') -> '1/nm^2':
        return 0.5 * alpha * (0.5 * alpha * (omega_p * qs[..., 0] + omega_m * qs[..., 1]) ** 2 - omega_p) * trial_f(qs)

    def trial_f_diff_1(qs: '[nm]') -> '1/nm^2':
        return 0.5 * alpha * (0.5 * alpha * (omega_m * qs[..., 0] + omega_p * qs[..., 1]) ** 2 - omega_p) * trial_f(qs)

    kwargs['trial_f'] = trial_f
    kwargs['trial_f_diffs'] = [trial_f_diff_0, trial_f_diff_1]

ho_pigs = PIGSIMM([mass, mass], [grid_range, grid_range], [grid_len, grid_len], total_potential, beta, num_links, **kwargs)

estimated_potential_energy = ho_pigs.expectation_value(total_potential) / KB  # K
estimated_total_energy = ho_pigs.energy_mixed / KB  # K
estimated_trace = ho_pigs.trace_renyi2

print('V = {} K'.format(estimated_potential_energy))
print('E_mixed = {} K'.format(estimated_total_energy))
print('trace = {}'.format(estimated_trace))


# Output wavefunction.
if wf_out:
    np.savetxt(wf_out, np.hstack((ho_pigs.grid, ho_pigs.ground_wf[:, np.newaxis])))

# Output plot.
if density_diagonal_out:
    from pathintmatmult.plotting import plot2d

    xy_range = (-grid_range, grid_range)
    density = ho_pigs.density_diagonal.reshape(grid_len, grid_len)

    plot2d(density, xy_range, xy_range, density_diagonal_out, x_label=r'$q_2 / \mathrm{nm}$', y_label=r'$q_1 / \mathrm{nm}$')
