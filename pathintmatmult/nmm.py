"""
Numerical matrix multiplication for path integrals.
"""

import numpy as N

from .constants import HBAR
from .tools import cached


class PIGSMM:
	"""
	Path Integral Ground State via Matrix Multiplication

	Calculate the approximate ground state wavefunction of a system comprised
	of a particle in an arbitrary potential on a discretized and truncated
	grid. The wavefunction is determined via imaginary time propagation from a
	trial function using numerical matrix multiplication.
	"""

	def __init__(self, mass: 'g/mol', grid_range: 'nm', grid_len: '1', beta: 'mol/kJ', num_links: '1', pot_f: 'nm -> kJ/mol'):
		"""
		Note:
		  The convention used is that beta represents the entire path, so the
		  propagation length from the trial function to the middle of the path
		  is beta/2.

		Note:
		  If pot_f receives an array as input, it should map over it.

		Parameters:
		  mass: Mass of the particle.
		  grid_range: Where the grid is truncated. The grid is symmetric about
		              the origin.
		  grid_len: How many points are on the grid.
		  beta: Propagation length of the entire path.
		  num_links: Number of links in the entire path.
		  pot_f: Potential experienced by the particle at some position in
		         space.
		"""

		assert mass >= 0, 'Mass must not be negative.'
		assert grid_range > 0, 'Grid must have a positive length.'
		assert grid_len >= 2, 'Grid must have at least two points.'
		assert beta > 0, 'Beta must be positive.'
		assert num_links > 0, 'Must have at least one link.'
		assert num_links % 2 == 0, 'Number of links must be even.'

		self._mass = mass
		self._grid_range = grid_range
		self._grid_len = grid_len
		self._beta = beta
		self._num_links = num_links
		self._pot_f = pot_f

		# For cached decorator.
		self._cached = {}

	@property
	def mass(self) -> 'g/mol':
		return self._mass

	@property
	def grid_range(self) -> 'nm':
		return self._grid_range

	@property
	def grid_len(self) -> '1':
		return self._grid_len

	@property
	def beta(self) -> 'mol/kJ':
		return self._beta

	@property
	def num_links(self) -> '1':
		return self._num_links

	@property
	def pot_f(self) -> 'nm -> kJ/mol':
		return self._pot_f

	@property
	@cached
	def tau(self) -> 'mol/kJ':
		"""
		High-temperature propagator length.
		"""

		return self.beta / self.num_links

	@property
	@cached
	def grid(self) -> '[nm]':
		"""
		Vector of the positions corresponding to the grid points.
		"""

		return N.linspace(-self.grid_range, self.grid_range, self.grid_len)

	@property
	@cached
	def grid_spacing(self) -> 'nm':
		"""
		Distance between adjacent grid points.
		"""

		return 2 * self.grid_range / (self.grid_len - 1)

	@property
	@cached
	def volume_element(self) -> 'nm':
		"""
		Effective volume taken up by each grid point.
		"""

		return self.grid_spacing

	@property
	@cached
	def pot_f_grid(self) -> '[kJ/mol]':
		"""
		Potential function evaluated on the grid.
		"""

		return self.pot_f(self.grid)

	@property
	@cached
	def trial_f_grid(self) -> '[1/nm^1/2]':
		"""
		Normalized trial function evaluated on the grid.

		Currently only has support for a uniform trial function.
		"""

		return N.ones(self.grid_len) / N.sqrt(self.grid_len)

	@property
	@cached
	def rho_tau(self) -> '[[1/nm]]':
		"""
		Matrix for the high-temperature propagator.
		"""

		prefactor_K = self.mass / (2 * HBAR * HBAR * self.tau) # 1/nm^2
		prefactor_V = self.tau / 2 # mol/kJ
		prefactor_front = N.sqrt(prefactor_K / N.pi) # 1/nm

		K = N.tile(self.grid, (self.grid_len, 1)) # [[nm]]
		V = N.tile(self.pot_f_grid, (self.grid_len, 1)) # [[kJ/mol]]

		return prefactor_front * N.exp(-prefactor_K * (K - K.T) ** 2 - prefactor_V * (V + V.T))

	@property
	@cached
	def rho_beta_half(self) -> '[[1/nm]]':
		"""
		Matrix for the half path propagator.
		"""

		power = self.num_links // 2

		eigvals, eigvecs = N.linalg.eigh(self.volume_element * self.rho_tau)
		result = N.dot(N.dot(eigvecs, N.diag(eigvals ** power)), eigvecs.T)

		return result / self.volume_element

	@property
	@cached
	def rho_beta(self) -> '[[1/nm]]':
		"""
		Matrix for the full path propagator.
		"""

		return self.volume_element * N.dot(self.rho_beta_half, self.rho_beta_half)

	@property
	@cached
	def ground_wf(self) -> '[1/nm^1/2]':
		"""
		Normalized ground state wavefunction.
		"""

		ground_wf = N.dot(self.rho_beta_half, self.trial_f_grid)
		# Explicitly normalize.
		ground_wf /= N.sqrt(N.sum(ground_wf ** 2))

		return ground_wf

	@property
	@cached
	def density(self) -> '[[1/nm]]':
		"""
		Normalized ground state density matrix.
		"""

		return N.outer(self.ground_wf, self.ground_wf)

	@property
	@cached
	def density_diagonal(self) -> '[1/nm]':
		"""
		Normalized ground state diagonal density.
		"""

		return self.ground_wf ** 2

	@property
	@cached
	def energy_mixed(self) -> 'kJ/mol':
		"""
		Ground state energy calculated using the mixed estimator.

		Currently only has support for a uniform trial function.
		"""

		ground_wf_full = N.dot(self.rho_beta, self.trial_f_grid)

		energy = N.sum(ground_wf_full * self.pot_f_grid * self.trial_f_grid)
		normalization = N.dot(ground_wf_full, self.trial_f_grid)

		return energy / normalization

	def expectation_value(self, property_f: '[nm] -> [X]') -> 'X':
		"""
		Ground state expectation value of property_f.
		"""

		return N.dot(self.density_diagonal, property_f(self.grid))


class PIGSMM2(PIGSMM):
	"""
	PIGSMM for two identical particles.
	"""

	# pot_f(self) -> '[nm] -> kJ/mol'

	@property
	@cached
	def grid(self) -> '[[nm]]':
		"""
		Vector of the positions corresponding to the grid points.

		Actually a 2xN array, with each row containing coordinates of both
		particles.
		"""

		single = N.linspace(-self.grid_range, self.grid_range, self.grid_len) # [nm]
		double = N.array([[a, b] for a in single for b in single]) # [[nm]]

		return double

	@property
	@cached
	def volume_element(self) -> 'nm^2':
		"""
		Effective volume taken up by each grid point.
		"""

		return self.grid_spacing ** 2

	@property
	@cached
	def trial_f_grid(self) -> '[1/nm]':
		"""
		Normalized trial function evaluated on the grid.

		Currently only has support for a uniform trial function.
		"""

		num_points = self.grid_len ** 2

		return N.ones(num_points) / N.sqrt(num_points)

	@property
	@cached
	def rho_tau(self) -> '[[1/nm^2]]':
		"""
		Matrix for the high-temperature propagator.
		"""

		prefactor_K = self.mass / (2 * HBAR * HBAR * self.tau) # 1/nm^2
		prefactor_V = self.tau / 2 # mol/kJ
		prefactor_front = prefactor_K / N.pi # 1/nm^2

		K = N.empty((self.grid_len ** 2, self.grid_len ** 2)) # nm^2
		V = N.empty_like(K) # kJ/mol

		for i, q_i in enumerate(self.grid):
			for j, q_j in enumerate(self.grid):
				K[i, j] = N.sum((q_i - q_j) ** 2)
				V[i, j] = self.pot_f(q_i) + self.pot_f(q_j)

		return prefactor_front * N.exp(-prefactor_K * K - prefactor_V * V)

	# rho_beta_half(self) -> '[[1/nm^2]]'

	# rho_beta(self) -> '[[1/nm^2]]'

	# ground_wf(self) -> '[1/nm]'

	# density(self) -> '[[1/nm^2]]'

	# density_diagonal(self) -> '[1/nm^2]'

	# expectation_value(self, property_f: '[[nm]] -> [X]') -> 'X'

	@property
	@cached
	def density_reduced(self) -> '[[1/nm]]':
		"""
		Density matrix for one particle, with the other traced out.
		"""

		density_new = N.zeros((self.grid_len, self.grid_len))

		for i in range(self.grid_len):
			for j in range(self.grid_len):
				for t in range(self.grid_len):
					density_new[i, j] += self.density[self.grid_len * i + t, self.grid_len * j + t]

		return density_new

	@property
	@cached
	def trace_renyi2(self) -> '1':
		"""
		Trace of the square of the reduced density matrix.

		The 2nd Rényi entropy is the negative logarithm of this quantity.
		"""

		return N.linalg.matrix_power(self.density_reduced, 2).trace()
