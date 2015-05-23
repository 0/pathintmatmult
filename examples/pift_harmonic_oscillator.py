#!/usr/bin/env python3

"""
Harmonic oscillator PIFT example.

An oscillator with an angular frequency of x kelvin at reciprocal temperature
beta reciprocal kelvin has a thermal potential energy (in kelvin) of

  (1/4) x coth(0.5 beta x)

and a total energy of twice that. For example, for an oscillator with an
angular frequency of 1 K, at 0.1 K the thermal averages are approximately
0.2500 K and 0.5000 K (very nearly the zero point energies), while at 10 K they
are approximately 5.0042 K and 10.008 K. By 100 K, the total energy is about
100.00 K, so we are effectively at the classical limit.
"""

from argparse import ArgumentParser

from pathintmatmult import PIFTMM
from pathintmatmult.constants import HBAR, KB, ME
from pathintmatmult.potentials import harmonic_potential


# Parse arguments.
p = ArgumentParser(description='Calculate HO thermal properties using PIFTMM.')
p_config = p.add_argument_group('configuration')

p_config.add_argument('--mass', metavar='M', type=float, required=True, help='particle mass (electron masses)')
p_config.add_argument('--omega', metavar='W', type=float, required=True, help='angular frequency (K)')
p_config.add_argument('--grid-range', metavar='R', type=float, required=True, help='grid range from origin (nm)')
p_config.add_argument('--grid-len', metavar='L', type=int, required=True, help='number of points on grid')
p_config.add_argument('--beta', metavar='B', type=float, required=True, help='reciprocal temperature (1/K)')
p_config.add_argument('--num-links', metavar='P', type=int, required=True, help='number of links')

p.add_argument('--density-out', metavar='FILE', help='path to output density plot')

args = p.parse_args()

mass = args.mass * ME  # g/mol
omega = args.omega * KB / HBAR  # 1/ps
grid_range = args.grid_range  # nm
grid_len = args.grid_len  # 1
beta = args.beta / KB  # mol/kJ
num_links = args.num_links  # 1

density_out = args.density_out


# Calculate values.
harmonic = harmonic_potential(m=mass, w=omega)

ho_pift = PIFTMM([mass], [grid_range], [grid_len], harmonic, beta, num_links)

estimated_potential_energy = ho_pift.expectation_value(harmonic) / KB  # K

print('V = {} K'.format(estimated_potential_energy))
# According to the virial theorem, <K> = <V> for a harmonic oscillator.
print('E_virial = {} K'.format(2 * estimated_potential_energy))


# Output plot.
if density_out:
    from pathintmatmult.plotting import plot2d

    xy_range = (-grid_range, grid_range)

    plot2d(ho_pift.density, xy_range, xy_range, density_out, x_label=r'$q_j / \mathrm{nm}$', y_label=r'$q_i / \mathrm{nm}$')
