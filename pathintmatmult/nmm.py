"""
Numerical matrix multiplication for path integrals.
"""

from itertools import product

import numpy as np

from .constants import HBAR
from .tools import cached


class PIMM:
    """
    Path Integrals via Matrix Multiplication

    Base class for various kinds of path integral implementations.
    """

    def __init__(self, masses: '[g/mol]', grid_ranges: '[nm]',
                 grid_lens: '[1]', pot_f: '[nm] -> kJ/mol',
                 beta: 'mol/kJ', num_links: '1'):
        """
        Note:
          When pot_f receives an N-dimensional array as input, it needs to map
          over it, returning an (N-1)-dimensional array.

        Note:
          The "particles" are actually any Cartesian degrees of freedom. One
          might have the same configuration (masses and grids) for a
          3-dimensional 1-particle system as for a 1-dimensional 3-particle
          system. Of course, the coordinate arrays must be interpreted
          appropriately in each case (whether by the potential function or by
          the user of the output density).

        Parameters:
          masses: Masses of the particles.
          grid_ranges: Where the grids are truncated. Each grid is symmetric
                       about the origin.
          grid_lens: How many points are on the grids.
          beta: Propagation length of the entire path.
          num_links: Number of links in the entire path.
          pot_f: Potential experienced by the particles in some spatial
                 configuration.
        """

        assert len(masses) == len(grid_ranges) == len(grid_lens), \
            'Numbers of configuration items must match.'
        assert all(m > 0 for m in masses), 'Masses must be positive.'
        assert all(gr > 0 for gr in grid_ranges), 'Grids must have positive lengths.'
        assert all(gl >= 2 for gl in grid_lens), 'Grids must have at least two points.'
        assert beta > 0, 'Beta must be positive.'
        assert num_links >= 2, 'Must have at least two links.'

        self._masses = np.array(masses)
        self._grid_ranges = np.array(grid_ranges)
        self._grid_lens = np.array(grid_lens)
        self._pot_f = pot_f
        self._beta = beta
        self._num_links = num_links

        # For cached decorator.
        self._cached = {}

    @property
    def masses(self) -> '[g/mol]':
        return self._masses

    @property
    def grid_ranges(self) -> '[nm]':
        return self._grid_ranges

    @property
    def grid_lens(self) -> '[1]':
        return self._grid_lens

    @property
    def pot_f(self) -> '[nm] -> kJ/mol':
        return self._pot_f

    @property
    def beta(self) -> 'mol/kJ':
        return self._beta

    @property
    def num_links(self) -> '1':
        return self._num_links

    @property
    @cached
    def tau(self) -> 'mol/kJ':
        """
        High-temperature propagator length.
        """

        return self.beta / self.num_links

    @property
    @cached
    def num_points(self) -> '1':
        """
        Number of points in the coordinate vector.
        """

        return np.prod(self.grid_lens)

    @property
    @cached
    def grid(self) -> '[[nm]]':
        """
        Vector of the positions corresponding to the grid points.

        This is not a vector in the sense of a 1-dimensional array, because
        each element is itself a vector of coordinates for each particle.
        However, it can be thought of as the tensor product of the
        1-dimensional position vectors.
        """

        grids = [np.linspace(-gr, gr, gl) for (gr, gl) in zip(self.grid_ranges, self.grid_lens)]
        result = np.array(list(product(*grids)))

        assert result.shape == (self.num_points, len(self.masses))

        return result

    @property
    @cached
    def volume_element(self) -> 'nm^N':
        """
        Effective volume taken up by each grid point.
        """

        return np.prod(self.grid_ranges / (self.grid_lens - 1))

    @property
    @cached
    def pot_f_grid(self) -> '[kJ/mol]':
        """
        Potential function evaluated on the grid.
        """

        return self.pot_f(self.grid)

    @property
    @cached
    def rho_tau(self) -> '[[1/nm^N]]':
        """
        Matrix for the high-temperature propagator.
        """

        prefactors_K = self.masses / (2 * HBAR * HBAR * self.tau)  # [1/nm^2]
        prefactor_V = self.tau / 2  # mol/kJ
        prefactor_front = np.sqrt(np.prod(prefactors_K) / np.pi)  # 1/nm^N

        K = np.empty((self.num_points, self.num_points))  # [[nm^2]]
        V = np.empty_like(K)  # [[kJ/mol]]

        for i, q_i in enumerate(self.grid):
            for j, q_j in enumerate(self.grid):
                K[i, j] = np.sum(prefactors_K * (q_i - q_j) ** 2)
                V[i, j] = self.pot_f(q_i) + self.pot_f(q_j)

        return prefactor_front * np.exp(-K - prefactor_V * V)

    @property
    def density_diagonal(self) -> '[1]':
        """
        Normalized diagonal density.
        """

        raise NotImplementedError()

    def expectation_value(self, property_f: '[nm] -> X') -> 'X':
        """
        Expectation value of property_f.

        Note:
          This is only implemented for properties that are diagonal in the
          position representation.

        Note:
          When property_f receives an N-dimensional array as input, it should
          behave in the same manner as pot_f.
        """

        return np.dot(self.density_diagonal, property_f(self.grid))


class PIFTMM(PIMM):
    """
    Path Integral at Finite Temperature via Matrix Multiplication

    Calculate the approximate thermal density matrix of a system comprised of
    one or more particles in an arbitrary potential on a discretized and
    truncated grid. The density matrix is determined via numerical matrix
    multiplication of high-temperature matrices.
    """

    @property
    @cached
    def rho_beta(self) -> '[[1/nm^N]]':
        """
        Matrix for the full path propagator.
        """

        power = self.num_links - 1

        eigvals, eigvecs = np.linalg.eigh(self.volume_element * self.rho_tau)
        result = np.dot(np.dot(eigvecs, np.diag(eigvals ** power)), eigvecs.T)

        return result / self.volume_element

    @property
    @cached
    def density(self) -> '[[1]]':
        """
        Normalized thermal density matrix.
        """

        density = self.rho_beta
        # Explicitly normalize.
        density /= density.diagonal().sum()

        return density

    @property
    @cached
    def density_diagonal(self) -> '[1]':
        """
        Normalized thermal diagonal density.
        """

        return self.density.diagonal()


class PIGSMM(PIMM):
    """
    Path Integral Ground State via Matrix Multiplication

    Calculate the approximate ground state wavefunction of a system comprised
    of one or more particles in an arbitrary potential on a discretized and
    truncated grid. The wavefunction is determined via imaginary time
    propagation from a trial function using numerical matrix multiplication.
    """

    def __init__(self, masses: '[g/mol]', grid_ranges: '[nm]',
                 grid_lens: '[1]', pot_f: '[nm] -> kJ/mol',
                 beta: 'mol/kJ', num_links: '1', *,
                 trial_f: '[nm] -> 1' = None,
                 trial_f_diffs: '[[nm] -> 1/nm^2]' = None):
        """
        See PIMM.__init__ for more details.

        Note:
          The convention used is that beta represents the entire path, so the
          propagation length from the trial function to the middle of the path
          is beta/2.

        Note:
          When trial_f receives an N-dimensional array as input, it should
          behave in the same manner as pot_f.

        Parameters:
          trial_f: Approximation to the ground state wavefunction. If none is
                   provided, a uniform trial function is used.
          trial_f_diffs: Second derivatives of trial_f. One function must be
                         specified for each particle.
        """

        super().__init__(masses, grid_ranges, grid_lens, pot_f, beta, num_links)

        assert num_links % 2 == 0, 'Number of links must be even.'

        if trial_f is not None:
            assert trial_f_diffs is not None, 'Derivatives must be provided.'
            assert len(trial_f_diffs) == len(masses), 'Number of derivatives must match.'

        self._trial_f = trial_f
        self._trial_f_diffs = trial_f_diffs

    @property
    def trial_f(self) -> '[nm] -> 1':
        return self._trial_f

    @property
    def trial_f_diffs(self) -> '[[nm] -> 1/nm^2]':
        return self._trial_f_diffs

    @property
    @cached
    def uniform_trial_f_grid(self) -> '[1]':
        """
        Unnormalized uniform trial function evaluated on the grid.
        """

        return np.ones(self.num_points)

    @property
    @cached
    def trial_f_grid(self) -> '[1]':
        """
        Unnormalized trial function evaluated on the grid.
        """

        if self.trial_f is None:
            # Default to a uniform trial function.
            return self.uniform_trial_f_grid

        return self.trial_f(self.grid)

    @property
    @cached
    def uniform_trial_f_diffs_grid(self) -> '[[1/nm^2]]':
        """
        Unnormalized uniform trial function derivatives evaluated on the grid.
        """

        return np.zeros(self.grid.T.shape)

    @property
    @cached
    def trial_f_diffs_grid(self) -> '[[1/nm^2]]':
        """
        Unnormalized trial function derivatives evaluated on the grid.
        """

        if self.trial_f is None:
            # Default to a uniform trial function.
            return self.uniform_trial_f_diffs_grid

        result = np.empty(self.grid.T.shape)

        for i, f in enumerate(self.trial_f_diffs):
            result[i] = f(self.grid)

        return result

    @property
    @cached
    def rho_beta_half(self) -> '[[1/nm^N]]':
        """
        Matrix for the half path propagator.
        """

        power = self.num_links // 2

        eigvals, eigvecs = np.linalg.eigh(self.volume_element * self.rho_tau)
        result = np.dot(np.dot(eigvecs, np.diag(eigvals ** power)), eigvecs.T)

        return result / self.volume_element

    @property
    @cached
    def rho_beta(self) -> '[[1/nm^N]]':
        """
        Matrix for the full path propagator.
        """

        return self.volume_element * np.dot(self.rho_beta_half, self.rho_beta_half)

    @property
    @cached
    def ground_wf(self) -> '[1]':
        """
        Normalized ground state wavefunction.
        """

        ground_wf = np.dot(self.rho_beta_half, self.trial_f_grid)
        # Explicitly normalize.
        ground_wf /= np.sqrt(np.sum(ground_wf ** 2))

        return ground_wf

    @property
    @cached
    def density(self) -> '[[1]]':
        """
        Normalized ground state density matrix.
        """

        return np.outer(self.ground_wf, self.ground_wf)

    @property
    @cached
    def density_diagonal(self) -> '[1]':
        """
        Normalized ground state diagonal density.
        """

        return self.ground_wf ** 2

    @property
    @cached
    def energy_mixed(self) -> 'kJ/mol':
        """
        Ground state energy calculated using the mixed estimator.
        """

        ground_wf_full = np.dot(self.rho_beta, self.trial_f_grid)  # [1/nm^N]
        trial_f_diffs = np.sum(self.trial_f_diffs_grid / self.masses[:, np.newaxis], axis=0)  # [mol/g nm^2]

        energy_V = np.sum(ground_wf_full * self.pot_f_grid * self.trial_f_grid)  # kJ/mol nm^N
        energy_K = np.dot(ground_wf_full, trial_f_diffs)  # mol/g nm^(N+2)
        normalization = np.dot(ground_wf_full, self.trial_f_grid)  # 1/nm^N

        return (energy_V - 0.5 * HBAR * HBAR * energy_K) / normalization

    @property
    @cached
    def density_reduced(self) -> '[[1]]':
        """
        Density matrix for the first particle, with the other traced out.

        Only implemented for two-particle systems.
        """

        assert len(self.masses) == 2

        new_len = self.grid_lens[0]
        other_len = self.grid_lens[1]
        density_new = np.zeros((new_len, new_len))

        for i in range(new_len):
            for j in range(new_len):
                for t in range(other_len):
                    density_new[i, j] += self.density[other_len * i + t, other_len * j + t]

        return density_new

    @property
    @cached
    def trace_renyi2(self) -> '1':
        """
        Trace of the square of the reduced density matrix.

        The 2nd Rényi entropy is the negative logarithm of this quantity.
        """

        return np.linalg.matrix_power(self.density_reduced, 2).trace()
