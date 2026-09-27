# Upstream integration — 2026-09-27

Merge upstream `869e489e` into `75600955` on `semi-implicit-stiffness`.
The 27 upstream commits add MeshFEMSparse assembly, direct full-DOF assembly,
templated/SIMD geometry, shared host/device sources and CPU/CUDA LBVH work.
Tight-Inclusion 1.1.0 and the mollified-Hessian PSD repair were already present.

## Fork contracts retained

- All normal energy, gradient, Hessian, Gauss-Newton and shape-derivative paths
  retain `stiffness_scale`; new assembly calls the weighted local derivatives.
  The mollified zero branch projects `weight * stiffness_scale * potential`.
- Parent-contribution records and the exact sparse displacement-map accessor
  survive the merge. Full-DOF assembly uses direct index remapping only for a
  selection map and otherwise retains the sparse-map fallback.
- Checked HashGrid/BruteForce accounting, unsupported-budget refusals and
  exception-safe candidate construction remain in place. The new CUDA LBVH
  vertex build overrides the base entry point, so it now explicitly checks
  `check_budget_supported()` before clearing state or allocating device memory.

The two textual conflicts preserve the weighted mollified scalar and the
random-number include used by the fork's derivative tests. Added regressions
cover unequal/zero stiffness scales through selection and non-selection
full-DOF assembly and MeshFEM versus triplet Hessians. CUDA LBVH is included
in the conditional unsupported-budget regression.

## Native validation

macOS arm64, AppleClang 21, CPU, SIMD and MeshFEMSparse ON, CUDA OFF:

- Optimized assertion-enabled build: Release with `-O2`, without `NDEBUG`.
- Entire default test selection: **339 passed, 5 skipped; 4,975,065 assertions
  passed**, random seed 1, four threads, exit 0.
- Skips are the existing assertion-build exclusions: repeated CCD, STQ all
  cases, SpatialHash build benchmark, bunny assembly-pattern reuse and the
  BruteForce branch of the full barrier derivative test.
- Test fixtures: `c7eba549d9a80d15569a013c473f0aff104ac44a`, isolated checkout.
- Modified test files formatted with clang-format 21.1.2. The upstream patch
  context whitespace and Windows batch-file CRLF endings are retained.

The initial test run lacked fixtures: building only `ipc_toolkit_tests` does
not build the separate download target. Its missing-file failures and abort
are retained in the evidence. Explicitly building the fixture target and
rerunning produced the result above.

CUDA compilation/execution and native Linux/Windows are not established by
these local results. No performance claim or physical certification is made.
PolyFEM consumer acceptance and the adopted pin are recorded separately in
the consumer repository after rebuilding it.

Evidence in the parent workspace: `outputs/ipc-upstream/20260927/`.
`PROGRESS.md` tracks the consumer work; `configure.log`, `build.log`,
`build-tests-final.log`, `configure-test-data.log`, `build-test-data.log` and
`tests-all-with-data.log` retain the successful native checks. The first
fixture-less run is `tests-all.log`. No Teseo or private scene was run.
