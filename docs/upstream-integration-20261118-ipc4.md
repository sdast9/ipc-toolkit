# IPC Toolkit upstream integration, IPC-4

Merge upstream `282d4830f0f2a48f741e46ecfc8bfb72ec2b50d5` into
`50041d99f90dea65a1f37f4a492c4380fcb54c44`. The only conflict was
`src/ipc/broad_phase/hash_grid.cpp`.

The upstream `detect_cell_pairs` helper is unchanged. Its monotonic
scan visits each common key once. Its bounded batches and disjoint
parallel tiles visit each eligible position pair once. Across two lists,
a key with `n0` and `n1` items has at most `n0 * n1` emissions. Within
one list the strict upper triangle bounds emissions by `n * (n - 1) / 2`.
Filtering, AABB tests and deduplication can only reduce those counts.

The existing opt-in guards run before calling the traversal. The
cross-list guard scans the original sorted items and sums checked
products. The same-list guard retains its checked unordered-pair sum.
The cell-item preflight, checked grid geometry, named failures and
per-detection statistics are retained. Tests cover both relation types,
crowded tiled cells, the batch boundary, predicate order and exact
budget admissions and refusals.

On the isolated macOS arm64 CPU build, the pinned formatter, native
build and Python binding smoke passed. The focused toolkit run passed
20 cases and 2,510 assertions. The full default
suite passed 466 cases and 5,811,140 assertions with
seed 1 and four threads. The same six private CCD inputs were unavailable
as in IPC-3. Evidence is under sibling `IPC-4/`, including
`toolkit-merge-audit.json`, `toolkit-build-source.patch` and
`toolkit-tests-run/receipt.json`. The frozen binaries are under
`bin/IPC-4-toolkit-validation/`.
