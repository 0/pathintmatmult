# pathintmatmult

> Path integrals via numerical matrix multiplication.

This is a Python 3 package for calculating density matrices in the position basis using path integrals discretized with a symmetric Trotter factorization.
While this is hardly useful on its own, it can be used to verify a different implementation of discretized path integrals (e.g. PIMC or PIMD), since it provides access to the expected systematic error due to a finite number of Trotter links.

All the output densities (and wavefunctions) are normalized by including the volume element (or its square root) in the value.
This makes the values dimensionless and with the property that the sums of their diagonal elements (or of the squares of their elements) sum to unity.


## Examples

### `PIFTMM`

Density matrix for a single particle in a harmonic oscillator at 1 K:
![harmonic oscillator density](https://0.github.io/pathintmatmult/examples/pift_ho_density.png)
Generated using:
```
python examples/pift_harmonic_oscillator.py --mass 1 --omega 1 --grid-range 180 --grid-len 500 --beta 1 --num-links 1024 --density-out pift_ho_density.png
```

### `PIGSIMM`

Square of wavefunction (diagonal density) for the ground state of two harmonically-interacting particles in a harmonic trap:
![harmonic oscillator density](https://0.github.io/pathintmatmult/examples/pigs_ho_ent_density_diagonal.png)
Generated using:
```
python examples/pigs_harmonic_oscillator_entangled.py --mass 1 --omega-0 1 --omega-int 1 --grid-range 100 --grid-len 100 --beta 6 --num-links 192 --density-diagonal-out pigs_ho_ent_density_diagonal.png
```

### `PIGSMM`

Density matrix for a single particle in a harmonic oscillator ground state:
![harmonic oscillator density](https://0.github.io/pathintmatmult/examples/pigs_ho_density.png)
Generated using:
```
python examples/pigs_harmonic_oscillator.py --mass 1 --omega 1 --grid-range 120 --grid-len 500 --beta 16 --num-links 1024 --density-out pigs_ho_density.png
```


## License

Provided under the terms of the MIT license.
See LICENSE.txt for more information.
