// Opt-in broad-phase resource budget (BroadPhaseBudget): the bounds are
// checked from the boxes / the sorted items before the buffers they bound
// are allocated, a violation throws BroadPhaseBudgetExceeded (not a
// std::runtime_error), Candidates::build leaves no partial candidate set,
// and a generous budget reproduces the unbudgeted result exactly.
#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/broad_phase/brute_force.hpp>
#include <ipc/broad_phase/create_broad_phase.hpp>
#include <ipc/broad_phase/hash_grid.hpp>
#include <ipc/candidates/candidates.hpp>
#include <ipc/collision_mesh.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <igl/edges.h>

#include <stdexcept>
#include <type_traits>

using namespace ipc;

namespace {

// Two n×n triangulated sheets over [0,1]², the upper one at z = gap; the
// first `movers` vertices of the upper sheet sweep by `disp` along (1,1,1).
void two_sheets(
    const int n,
    const double gap,
    const int movers,
    const double disp,
    Eigen::MatrixXd& V0,
    Eigen::MatrixXd& V1,
    Eigen::MatrixXi& E,
    Eigen::MatrixXi& F)
{
    const int per_sheet = n * n;
    V0.resize(2 * per_sheet, 3);
    F.resize(4 * (n - 1) * (n - 1), 3);
    int f = 0;
    for (int sheet = 0; sheet < 2; ++sheet) {
        const int offset = sheet * per_sheet;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                V0.row(offset + i * n + j) << double(i) / (n - 1),
                    double(j) / (n - 1), sheet * gap;
            }
        }
        for (int i = 0; i + 1 < n; ++i) {
            for (int j = 0; j + 1 < n; ++j) {
                const int a = offset + i * n + j, b = a + 1, c = a + n,
                          d = c + 1;
                F.row(f++) << a, b, d;
                F.row(f++) << a, d, c;
            }
        }
    }
    igl::edges(F, E);
    V1 = V0;
    for (int i = 0; i < movers; ++i) {
        V1.row(per_sheet + i) += disp * Eigen::RowVector3d(1, 1, 1) / sqrt(3.);
    }
}

// The items HashGrid::insert_box emits, replicated from the boxes alone.
struct ItemEstimate {
    double cell_size;
    Eigen::Array3i grid;
    size_t items;
};

ItemEstimate estimate_items(
    const Eigen::MatrixXd& V0,
    const Eigen::MatrixXd& V1,
    const Eigen::MatrixXi& E,
    const Eigen::MatrixXi& F,
    const double inflation)
{
    AABBs vb, eb, fb;
    build_vertex_boxes(V0, V1, vb, inflation);
    build_edge_boxes(vb, E, eb);
    build_face_boxes(vb, F, fb);
    Eigen::Array3d lo = Eigen::Array3d::Constant(1e300),
                   hi = Eigen::Array3d::Constant(-1e300);
    for (const auto& b : vb) {
        lo = lo.min(b.min);
        hi = hi.max(b.max);
    }
    std::vector<double> extents;
    for (const auto& b : eb) {
        extents.push_back((b.max - b.min).maxCoeff());
    }
    std::sort(extents.begin(), extents.end());
    const size_t m = extents.size() / 2;
    ItemEstimate est;
    est.cell_size = extents.size() % 2 == 0
        ? 0.5 * (extents[m] + extents[m - 1])
        : extents[m];
    est.grid = ((hi - lo) / est.cell_size).ceil().cast<int>().max(1);
    est.items = 0;
    for (const AABBs* boxes : { &vb, &eb, &fb }) {
        for (const auto& b : *boxes) {
            Eigen::Array3i a = ((b.min - lo) / est.cell_size).cast<int>();
            Eigen::Array3i z = ((b.max - lo) / est.cell_size).cast<int>();
            a = a.max(0).min(est.grid - 1);
            z = z.max(0).min(est.grid - 1);
            est.items += size_t(z.x() - a.x() + 1) * size_t(z.y() - a.y() + 1)
                * size_t(z.z() - a.z() + 1);
        }
    }
    return est;
}

} // namespace

TEST_CASE(
    "Broad-phase budget: exception type and unsupported methods",
    "[broad_phase][budget]")
{
    static_assert(
        !std::is_base_of_v<std::runtime_error, BroadPhaseBudgetExceeded>);
    static_assert(std::is_base_of_v<std::exception, BroadPhaseBudgetExceeded>);

    CHECK(!BroadPhaseBudget().enabled());
    CHECK(HashGrid().supports_budget());
    CHECK(BruteForce().supports_budget());
    for (const auto method :
         { BroadPhaseMethod::LBVH, BroadPhaseMethod::SPATIAL_HASH,
           BroadPhaseMethod::SWEEP_AND_PRUNE }) {
        auto bp = create_broad_phase(method);
        CHECK(!bp->supports_budget());
        bp->budget.max_cell_items = 10;
        CHECK_THROWS_AS(bp->check_budget_supported(), std::invalid_argument);
        Eigen::MatrixXd V0, V1;
        Eigen::MatrixXi E, F;
        two_sheets(4, .05, 1, .5, V0, V1, E, F);
        Candidates candidates;
        CHECK_THROWS_AS(
            candidates.build(CollisionMesh(V0, E, F), V0, V1, 1e-3, bp.get()),
            std::invalid_argument);
        CHECK(candidates.empty());
        // Also on the direct build entry points (some methods override them).
        CHECK_THROWS_AS(bp->build(V0, E, F, 1e-3), std::invalid_argument);
        CHECK_THROWS_AS(bp->build(V0, V1, E, F, 1e-3), std::invalid_argument);
        bp->budget = BroadPhaseBudget();
        CHECK_NOTHROW(bp->check_budget_supported());
    }
}

TEST_CASE(
    "Hash-grid budget fires before allocation, leaves no partial set, and is exact",
    "[broad_phase][budget]")
{
    Eigen::MatrixXd V0, V1;
    Eigen::MatrixXi E, F;
    two_sheets(
        10, .05, 1, 2., V0, V1, E,
        F); // one vertex sweeps 2 units: its boxes cover many cells
    const CollisionMesh mesh(V0, E, F);
    const double inflation = 5e-4;
    const ItemEstimate est = estimate_items(V0, V1, E, F, inflation);
    REQUIRE(est.items > 2 * size_t(V0.rows() + E.rows() + F.rows()));

    HashGrid unlimited;
    Candidates reference;
    reference.build(mesh, V0, V1, inflation, &unlimited);
    REQUIRE(!reference.empty());
    const BroadPhaseBuildStatistics& measured = unlimited.build_statistics();
    CHECK(measured.measured);
    CHECK(measured.cell_items == est.items);
    CHECK(measured.cell_size == est.cell_size);
    CHECK(measured.grid_size[0] == est.grid[0]);
    CHECK(measured.grid_size[1] == est.grid[1]);
    CHECK(measured.grid_size[2] == est.grid[2]);
    CHECK(measured.candidate_emissions == 0); // counted only under a budget

    SECTION("item bound just below the exact count")
    {
        HashGrid grid;
        grid.budget.max_cell_items = est.items - 1;
        Candidates candidates;
        try {
            candidates.build(mesh, V0, V1, inflation, &grid);
            FAIL("no exception");
        } catch (const BroadPhaseBudgetExceeded& e) {
            CHECK(e.method == "HashGrid");
            CHECK(e.quantity == "cell_items");
            CHECK(e.requested == est.items);
            CHECK(e.limit == est.items - 1);
            CHECK_THAT(
                e.what(),
                Catch::Matchers::ContainsSubstring("before allocation"));
        }
        CHECK(candidates.empty());
        CHECK(!grid.build_statistics().measured); // nothing was inserted
        // Just at the count: builds, and the result is the unbudgeted one.
        grid.budget.max_cell_items = est.items;
        candidates.build(mesh, V0, V1, inflation, &grid);
        CHECK(candidates.size() == reference.size());
        CHECK(grid.build_statistics().cell_items == est.items);
        CHECK(grid.build_statistics().candidate_emissions > 0);
    }
    SECTION("emission bound fires after the items and before the candidates")
    {
        HashGrid grid;
        grid.budget.max_cell_items = est.items; // allow the items
        grid.budget.max_candidate_emissions = 1;
        Candidates candidates;
        try {
            candidates.build(mesh, V0, V1, inflation, &grid);
            FAIL("no exception");
        } catch (const BroadPhaseBudgetExceeded& e) {
            CHECK(e.quantity == "candidate_emissions");
            CHECK(e.requested > 1);
            CHECK(e.limit == 1);
        }
        CHECK(candidates.empty());
        CHECK(grid.build_statistics().cell_items == est.items);
        // The exact per-call emission count admits the build.
        HashGrid counting;
        counting.budget.max_cell_items = est.items;
        counting.budget.max_candidate_emissions = size_t(-1) / 2;
        candidates.build(mesh, V0, V1, inflation, &counting);
        const size_t total = counting.build_statistics().candidate_emissions;
        REQUIRE(total > 0);
        // total is the sum of the EE and FV passes; the larger pass sets the
        // smallest admissible per-call bound. Bound at total: passes.
        counting.budget.max_candidate_emissions = total;
        candidates.build(mesh, V0, V1, inflation, &counting);
        CHECK(candidates.size() == reference.size());
        // Final candidates never exceed the pre-filter emissions.
        CHECK(candidates.size() <= total);
    }
    SECTION("repeat invocation after a failure")
    {
        HashGrid grid;
        grid.budget.max_cell_items = 1;
        Candidates candidates;
        CHECK_THROWS_AS(
            candidates.build(mesh, V0, V1, inflation, &grid),
            BroadPhaseBudgetExceeded);
        CHECK_THROWS_AS(
            candidates.build(mesh, V0, V1, inflation, &grid),
            BroadPhaseBudgetExceeded);
        grid.budget = BroadPhaseBudget();
        candidates.build(mesh, V0, V1, inflation, &grid);
        CHECK(candidates.size() == reference.size());
        std::vector<std::array<index_t, 4>> a, b;
        for (size_t i = 0; i < candidates.size(); ++i) {
            a.push_back(candidates[i].vertex_ids(E, F));
            b.push_back(reference[i].vertex_ids(E, F));
        }
        CHECK(a == b);
    }
}

TEST_CASE(
    "Brute-force budget bounds the box pairs compared", "[broad_phase][budget]")
{
    Eigen::MatrixXd V0, V1;
    Eigen::MatrixXi E, F;
    two_sheets(6, .05, 0, 0., V0, V1, E, F);
    const CollisionMesh mesh(V0, E, F);
    BruteForce bf;
    bf.budget.max_candidate_emissions =
        size_t(E.rows()) * (E.rows() - 1) / 2 - 1; // EE pairs minus one
    Candidates candidates;
    try {
        candidates.build(mesh, V0, V1, 1e-3, &bf);
        FAIL("no exception");
    } catch (const BroadPhaseBudgetExceeded& e) {
        CHECK(e.method == "BruteForce");
        CHECK(e.quantity == "candidate_emissions");
        CHECK(e.requested == size_t(E.rows()) * (E.rows() - 1) / 2);
    }
    CHECK(candidates.empty());
    bf.budget.max_candidate_emissions = size_t(E.rows()) * (E.rows() - 1) / 2;
    // The face-vertex pass compares |F| x |V| boxes: below the EE count here.
    REQUIRE(size_t(F.rows()) * V0.rows() <= bf.budget.max_candidate_emissions);
    candidates.build(mesh, V0, V1, 1e-3, &bf);
    BruteForce unlimited;
    Candidates reference;
    reference.build(mesh, V0, V1, 1e-3, &unlimited);
    CHECK(candidates.size() == reference.size());
    CHECK(
        bf.build_statistics().candidate_emissions
        == size_t(E.rows()) * (E.rows() - 1) / 2
            + size_t(F.rows()) * V0.rows());
}
