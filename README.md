# MinLpE
Adaptive sampling based on the minimization of approximation error Lp norm. MaxLpE is a fast adaptive sampling algorithm with accuracy comparable with the best known adaptive sampling methods: TEAD, LIP, MIPT, EIGF, MASA, SFVCT. Its features include smapling time control and parallel (batch) point generation. The norm parameter p regulates sampling, inclining it either towards local exploitation or, conversely, global exploration. Small Lp norm is achieved by reducing the function approximation error and the size of the region with large variation after adding a new sampling point. This solution is similar to kriging in terms of the choice between local exploitation and global exploration. The difference is in error estimation which depends on the values of the function in sampling points in contrast to homoscedastic variance estimate of kriging.

![Demo](./demo.gif)

## Installation

pip install --upgrade minlpe

## Usage

See file examples.py

### If you like the software acknowledge it using the references below:

[A.S.Algasov, S.A.Guda, V.I.Kolesnikov, V.V.Ilicheva, A.V.Soldatov. Fast adaptive sampling with operation time control // Journal of Computational Science, Volume 67, 2023, 101946, ISSN 1877-7503, https://doi.org/10.1016/j.jocs.2023.101946.](https://www.sciencedirect.com/science/article/abs/pii/S1877750323000066?via%3Dihub)
