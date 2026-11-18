# IPC Toolkit upstream integration, IPC-5

Merge `7d668025590e20adf58968d7d17104bb7a739a71` into the tested
IPC-4 toolkit `4f99f337357b366daf0984ea9c775273403e7e46`. The sole
conflict was `src/ipc/candidates/candidates.cpp`. Both static and swept
codimensional passes now translate subset positions to mesh vertex IDs
before filtering. Scoped restoration keeps those temporary references
inside each pass and restores the original filter on successful or
exceptional exits. Checked budgets, build statistics, candidate
cleanup and the all-types path carry forward.

The new focused regression hits both codimensional budget refusals in
static and swept builds. It checks the named exception, empty result,
restored filter, and successful retry. On the isolated macOS arm64 CPU
build, the formatter, standalone Eigen ABI check and Python binding
smoke passed. The focused suite passed 18 cases and
377 assertions. The full default suite passed
482 cases and 5,811,532 assertions on seed 1 and
four threads. Six private CCD fixtures were unavailable in both IPC-4
and IPC-5. Evidence is in sibling `IPC-5/`.

The consumer evaluation compares both IPC Toolkit SIMD settings.
With SIMD enabled, the IPC Toolkit CMake target pins
`EIGEN_DONT_VECTORIZE` on the shared Eigen target. The SIMD-off build
provides the alternative for the same source. The owner decides the
configuration after the comparison.
