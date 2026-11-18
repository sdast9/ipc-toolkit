# IPC-2 candidate and stencil integration (2026-11-17)

Merge upstream `c7ada28baa1ed875ad361f9404c2f7a0ee950efb` from IPC-1
`eec0ff82d2b7a0a9ce27fff0afd7de2c18e57d3a` on the isolated
`upstream-merge` branch.

- `broad_phase.hpp` keeps checked counting and includes upstream's
  `CandidateVector`. `brute_force.cpp` keeps the pre-allocation budget
  calculation and its named refusal with upstream's thread-local storage.
- Fork parent contributions, both collision flags, and selection and
  displacement assembly paths carry forward. The two retained fork budget
  test files use `CandidateVector` and `Candidates::visit` to exercise the
  same limits, ordered vertex IDs, and recovery assertions.
- The former compiled collision stencil source and documentation logo
  deletions come from upstream and had no fork changes. The audit retained
  all 114 prior named cases in the changed native test files.

The native macOS arm64 build and Python extension built with exit 0.
The full native suite on seed 1 with four threads passed 453 cases and
5,802,571 assertions. The isolated Python 3.9 binding test verified both
fork flags in all four combinations and equal vertex-pair distances and
coefficients through the candidate and normal-collision stencil. The
original compilation failure in the two budget test files is retained.

Evidence is under sibling `IPC-2/` in `toolkit-build-run/`,
`toolkit-build-recheck-run/` and `toolkit-tests-run/`. The validated tracked
input is `toolkit-recheck-source.patch`. Executable and binding digests
are in `bin/IPC-2-toolkit-validation/binary.sha256`. PolyFEM consumer
validation and publication are separate gates.
