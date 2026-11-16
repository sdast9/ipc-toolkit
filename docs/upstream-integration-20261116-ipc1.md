# IPC-1: merge through dff05854, 2026-11-16

Isolated `upstream-merge` integration branch, merge parents:
- Fork `semi-implicit-stiffness`: 7b8fcac7abd6801769e57166dc9fffb9bbcff5cc
- Upstream: dff05854d74617f1e21ccb2bb321c45d7b4c6802

No shared checkout, PolyFEM build, adopted pin, remeshing baseline or release branch was changed.

## Preservation review

Resolved the nine expected conflicts without dropping either side's behavior:
- `separate_signed_contributions` and `skip_obstacles` are independent members and constructor arguments, fork-first, both default OFF. Both are exposed as Python properties.
- Parent-contribution records and per-contact stiffness remain. Added combined improved-max/obstacle tests exercise actual negative parents, sign grouping and equal total unit-stiffness energy before/after separation.
- Candidate `all_types` propagates through the exception-safe build wrapper; statistics accumulation, resource checks and clear-on-failure remain. Added all-types HashGrid/BruteForce budget coverage (including resetting `has_all_types` on failure).
- Canonical collision ordering survives the smooth-contact to GCP rename. Fork determinism test retained and migrated to the new symbols; existing compatibility aliases remain upstream's.
- Both sides' CMake entries/tests retained. A source-name inventory found no prior TEST_CASE name missing. Independent read-only review found no concrete regression in these merge resolutions.
- Deletion audit: the only non-rename deleted file is `src/ipc/utils/unordered_map_and_set.cpp`, removed upstream and untouched by the fork since the merge base. No fork capability was intentionally removed.

## Verification

macOS arm64, AppleClang 21, RelWithDebInfo, CPU, SIMD and MeshFEMSparse ON, CUDA OFF; four-thread tests, random seed 1. Test fixtures cloned at upstream recipe pin 05696a21b8662f6cd12f8469eeb35ee9a5cc0493 before testing.

- Native toolkit/test build passed.
- Entire default toolkit suite: **453 cases passed, 5,802,589 assertions passed**, exit 0, about 88 seconds. This is the default non-hidden selection, not hidden benchmark execution or CUDA validation. Unlike the previous assertion-enabled record, this optimized build defines NDEBUG; the suite did not report skips.
- New combined interaction selection: **2 cases, 121 assertions passed**.
- Python module compiled successfully. Import and independent flag toggle checks passed in an isolated Python 3.9 venv with the package's declared numpy/scipy dependencies. No system Python packages changed. This is a focused binding smoke check, not the complete Python suite.
- Modified files formatted with repository-pinned clang-format 21.1.2; no hook bypass.

Initial diagnostic failures are preserved: my new test incorrectly expected opt-in BruteForce statistics with budgeting disabled; corrected the test to enable its budget rather than changing production behavior. A subsequent new-test compilation typo (unqualified nested enum) was corrected. A stale-binary test attempt after that compilation failure is explicitly marked and is not acceptance evidence. Initial Python imports lacked numpy/scipy; dependencies were installed only into the isolated validation venv. The final suite and binding checks above are the acceptance evidence.

Evidence in sibling `IPC-1/`: build/configure logs and receipts; tests-all-final.log and tests-all-final.receipt.json; tests-interaction.log; build-python.log/exit; python-flags-test-complete.log/exit; fork-test-name-audit.json. RM-0 evidence and prototype remain unchanged in their separate directory and branch.

IPC-1 meets its native toolkit gate. PolyFEM consumer validation is still required in PF-4; this result is not a claim that its upstream pin, all remeshing paths, CUDA, Linux or Windows are validated. Integration-only publication is authorized by the merge plan; release remains gated by REL-1.
