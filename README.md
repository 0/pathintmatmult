# pathintmatmult

> Path integrals via numerical matrix multiplication.

This is a Python 3 package for calculating density matrices in the position basis using path integrals discretized with a symmetric Trotter factorization.
While this is hardly useful on its own, it can be used to verify a different implementation of discretized path integrals (e.g. PIMC or PIMD), since it provides access to the expected systematic error due to a finite number of Trotter links.

Currently incredibly limited.
It supports only PIGS with one particle or with two identical particles, and uses a naïve approach that involves constructing the entire high-temperature propagator matrix.


## License

Provided under the terms of the MIT license.
See LICENSE.txt for more information.
