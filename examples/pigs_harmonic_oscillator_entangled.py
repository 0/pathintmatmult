#!/usr/bin/env python3

"""
Entangled harmonic oscillators PIGS example.

A pair of identical harmonic oscillators with a harmonic interaction potential.
"""

from argparse import ArgumentParser

from pathintmatmult import PIGSMM2
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

args = p.parse_args()

mass = args.mass * ME  # g/mol
omega_0 = args.omega_0 * KB / HBAR  # 1/ps
omega_int = args.omega_int * KB / HBAR  # 1/ps
grid_range = args.grid_range  # nm
grid_len = args.grid_len  # 1
beta = args.beta / KB  # mol/kJ
num_links = args.num_links  # 1


# Calculate values.
pot_0 = harmonic_potential(m=mass, w=omega_0)
pot_int = harmonic_potential(m=mass, w=omega_int)


def total_potential(qs):
    return pot_0(qs[..., 0]) + pot_0(qs[..., 1]) + pot_int(qs[..., 1] - qs[..., 0])

ho_pigs = PIGSMM2(mass, grid_range, grid_len, beta, num_links, total_potential)

estimated_potential_energy = ho_pigs.expectation_value(total_potential) / KB  # K
estimated_total_energy = ho_pigs.energy_mixed / KB  # K
estimated_trace = ho_pigs.trace_renyi2

print('V = {} K'.format(estimated_potential_energy))
print('E_mixed = {} K'.format(estimated_total_energy))
print('trace = {}'.format(estimated_trace))
