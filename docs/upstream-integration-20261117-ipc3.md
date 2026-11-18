# IPC-3 per-primitive thickness integration (2026-11-17)

Merge upstream `54d58ebcf1f6dd0cf95bcc58afd79839ca603a99` from IPC-2
`8854a705370d3cdd933748268955ca611f0c9491` on the isolated
`upstream-merge` branch.

- The static and swept candidate builds keep the fork's budget preflight,
  cleanup after partial builds and aggregate statistics. Their broad-phase
  calls use upstream's thickness-aware boxes and collision filters.
- Normal collision construction keeps the fork's `parent_of` and signed
  parent records alongside upstream's thickness helpers. The legacy scalar
  `dmin` remains an explicit override.
- Both frozen Python extensions were exercised with vertex-vertex,
  edge-vertex, edge-edge, face-vertex and plane-vertex fixtures, in standard
  IPC and area-weighted improved max. Omitted, explicit-zero and positive
  scalar settings all retained the exact stored minima and weights.
  The IPC-3 cases matched IPC-2 in both absent and explicit-zero thickness
  representations, totaling 60 matched collision records.

The native toolkit build and Python extension built with exit 0. The
full native default suite on seed 1 and four threads passed 462 cases and
5,808,918 assertions. The focused Python binding smoke passed both fork
flag properties and candidate-to-stencil parity. The 12 absent private
CCD fixture warnings match the IPC-2 baseline.

Evidence is under sibling `IPC-3/` in `toolkit-build-source.patch`,
`toolkit-build-run/`, `minimum-distance-parity.json` and
`toolkit-tests-run/`. Frozen binary digests are in
`bin/IPC-3-toolkit-validation/binary.sha256`. PolyFEM consumer validation
and publication are separate gates.
