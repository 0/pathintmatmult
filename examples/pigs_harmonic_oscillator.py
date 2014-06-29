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
"""

from argparse import ArgumentParser

from pathintmatmult.constants import HBAR, KB, ME
from pathintmatmult.nmm import PIGSMM
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

args = p.parse_args()

mass = args.mass * ME # g/mol
omega = args.omega * KB / HBAR # 1/ps
grid_range = args.grid_range # nm
grid_len = args.grid_len # 1
beta = args.beta / KB # mol/kJ
num_links = args.num_links # 1


# Calculate values.
harmonic = harmonic_potential(m=mass, w=omega)
ho_pigs = PIGSMM(mass, grid_range, grid_len, beta, num_links, harmonic)

estimated_potential_energy = ho_pigs.expectation_value(harmonic) / KB # K
estimated_total_energy = ho_pigs.energy_mixed / KB # K

print('V = {} K'.format(estimated_potential_energy))
# According to the virial theorem, <K> = <V> for a harmonic oscillator.
print('E_virial = {} K'.format(2 * estimated_potential_energy))
print('E_mixed = {} K'.format(estimated_total_energy))