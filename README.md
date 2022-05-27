# MinLpE
Adaptive sampling based on the minimization of approximation error Lp norm. MaxLpE is a fast adaptive sampling algorithm with accuracy comparable with the best known adaptive sampling methods: TEAD, LIP, MIPT, EIGF, MASA, SFVCT. Its features include smapling time control and parallel (batch) point generation. The norm parameter p regulates sampling, inclining it either towards local exploitation or, conversely, global exploration. Small Lp norm is achieved by reducing the function approximation error and the size of the region with large variation after adding a new sampling point. This solution is similar to kriging in terms of the choice between local exploitation and global exploration. The difference is in error estimation which depends on the values of the function in sampling points in contrast to homoscedastic variance estimate of kriging.

![Demo](./demo.gif)

### If you like the software acknowledge it using the references below:

S.A.Guda, A.S.Algasov, V.I.Kolesnikov, A.V.Soldatov, V.V.Ilicheva Fast adaptive sampling with operation time control // Journal of Computational Science, in press.
