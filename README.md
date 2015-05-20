# pathintmatmult

> Path integrals via numerical matrix multiplication.

This is a Python 3 package for calculating density matrices in the position basis using path integrals discretized with a symmetric Trotter factorization.
While this is hardly useful on its own, it can be used to verify a different implementation of discretized path integrals (e.g. PIMC or PIMD), since it provides access to the expected systematic error due to a finite number of Trotter links.

Currently incredibly limited.
It supports only PIGS with one particle or with two identical particles, and uses a naïve approach that involves constructing the entire high-temperature propagator matrix.

All the output densities (and wavefunctions) are normalized by including the volume element (or its square root) in the value.
This makes the values dimensionless and with the property that the sums of their diagonal elements (or of the squares of their elements) sum to unity.


## Example

An example density matrix for a single particle in a harmonic oscillator ground state:
![harmonic oscillator density](https://0.github.io/pathintmatmult/examples/density.png)
Generated using:
```
python examples/pigs_harmonic_oscillator.py --mass 1 --omega 1 --grid-range 120 --grid-len 500 --beta 16 --num-links 1024 --density-out density.png
```


## License

Provided under the terms of the MIT license.
See LICENSE.txt for more information.
