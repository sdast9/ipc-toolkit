// Checked arithmetic in the broad-phase resource accounting (RBR-02 of the
// RB-05 budget): a count that does not fit size_t can never wrap into an
// apparently cheap build, a reported count is exact or says "at least", a
// grid the hash key cannot index is refused by name before any conversion,
// and non-finite geometry never reaches a floating-point-to-int cast.
// Count-only throughout: no cell insertion or large allocation happens.
#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/broad_phase/brute_force.hpp>
#include <ipc/broad_phase/checked_count.hpp>
#include <ipc/broad_phase/hash_grid.hpp>
#include <ipc/candidates/candidates.hpp>
#include <ipc/collision_mesh.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <tbb/global_control.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace ipc;

namespace {

constexpr size_t SIZE_MAX_ = std::numeric_limits<size_t>::max();
static_assert(SIZE_MAX_ == 18446744073709551615ull, "64-bit size_t assumed");
// The hash key is a long: 64 bits on LP64 platforms, 32 on LLP64 (Windows).
// The grids below that approach 2^63 cells are representable only with a
// 64-bit key; the checked arithmetic itself is tested on every platform.
constexpr bool KEY_IS_64_BIT = sizeof(long) == 8;

// Exposes the protected count/check/resize methods; boxes are set directly.
class CountProbe : public HashGrid {
public:
    // A 3D grid of n cells per axis (cell size 1) and `boxes` boxes that
    // each cover the whole grid.
    void set_up(const Eigen::Array3i& n, const int boxes)
    {
        clear();
        dim = 3;
        resize(Eigen::Array3d::Zero(), n.cast<double>(), 1.0);
        AABB box;
        box.min = Eigen::Array3d::Zero();
        box.max = (n - 1).cast<double>();
        for (int i = 0; i < boxes; ++i) {
            vertex_boxes.push_back(box);
        }
    }
    void add_unit_boxes(const int count)
    {
        AABB box;
        box.min = Eigen::Array3d::Zero();
        box.max = Eigen::Array3d::Constant(0.5);
        for (int i = 0; i < count; ++i) {
            vertex_boxes.push_back(box);
        }
    }
    CheckedCount count() const { return count_cell_items(vertex_boxes); }
    void check() const { check_cell_item_budget(); }
    void resize_public(
        const Eigen::Array3d& lo, const Eigen::Array3d& hi, const double cell)
    {
        clear();
        dim = 3;
        resize(lo, hi, cell);
    }
    void
    range_public(const AABB& box, Eigen::Array3i& lo, Eigen::Array3i& hi) const
    {
        box_cell_range(box, lo, hi);
    }
    // Synthetic sorted items for the emission counts (the boxes they refer
    // to are disjoint, so no candidate survives the AABB test).
    void set_items(
        const std::vector<std::pair<long, int>>& vertex_key_counts,
        const std::vector<std::pair<long, int>>& edge_key_counts)
    {
        clear();
        dim = 3;
        resize(Eigen::Array3d::Zero(), Eigen::Array3d::Constant(1e3), 1.0);
        const auto fill = [&](const std::vector<std::pair<long, int>>& kc,
                              std::vector<HashItem>& items, AABBs& boxes,
                              const bool edges) {
            long id = 0;
            for (const auto& [key, n] : kc) {
                for (int i = 0; i < n; ++i, ++id) {
                    items.emplace_back(key, id);
                    // Disjoint boxes (the edge ones offset by 1) so that no
                    // candidate survives the AABB test.
                    AABB box;
                    box.min = Eigen::Array3d::Constant(
                        2.0 * id + (edges ? 1.0 : 0.0));
                    box.max = box.min + 0.5;
                    // Distinct vertex ids so no pair shares an endpoint.
                    box.vertex_ids = edges
                        ? std::array<index_t, 3> { { index_t(1000 + 2 * id),
                                                     index_t(1001 + 2 * id),
                                                     -1 } }
                        : std::array<index_t, 3> { { index_t(id), -1, -1 } };
                    boxes.push_back(box);
                }
            }
        };
        fill(vertex_key_counts, vertex_items, vertex_boxes, false);
        fill(edge_key_counts, edge_items, edge_boxes, true);
    }
};

constexpr size_t two_pow(const int k) { return size_t(1) << k; }

} // namespace

TEST_CASE("CheckedCount: products", "[broad_phase][budget][checked_count]")
{
    static_assert(std::is_trivially_copyable_v<CheckedCount>);
    CHECK(CheckedCount::product(0, SIZE_MAX_) == CheckedCount(0));
    CHECK(CheckedCount::product(SIZE_MAX_, 0) == CheckedCount(0));
    CHECK(CheckedCount::product(1, SIZE_MAX_) == CheckedCount(SIZE_MAX_));
    CHECK(CheckedCount::product(SIZE_MAX_, 1).exact());
    CHECK(
        CheckedCount::product(two_pow(32), two_pow(32) - 1)
        == CheckedCount(18446744069414584320ull)); // 2^64 - 2^32, fits
    const CheckedCount over = CheckedCount::product(two_pow(32), two_pow(32));
    CHECK(over.overflowed);
    CHECK(over.value == SIZE_MAX_);
    CHECK(!over.exact());
    CHECK(CheckedCount::product(SIZE_MAX_, 2).overflowed);
    // The review's grid: 2^22 * 2^21 * 2^21 = 2^64 wrapped to zero.
    const CheckedCount review =
        CheckedCount::product(two_pow(22), two_pow(21), two_pow(21));
    CHECK(review.overflowed);
    CHECK(review.value == SIZE_MAX_);
    // Largest representable 3D grid used below: 2^63 - 2^42.
    CHECK(
        CheckedCount::product(two_pow(21), two_pow(21), two_pow(21) - 1)
        == CheckedCount(9223367638808264704ull));
    // times(): an overflowed count stays overflowed unless multiplied by 0.
    CHECK(over.times(1).overflowed);
    CHECK(over.times(0) == CheckedCount(0));
    CHECK(CheckedCount(3).times(5) == CheckedCount(15));
    CHECK(CheckedCount(two_pow(63)).times(2).overflowed);
    CHECK(CheckedCount(two_pow(62)).times(3).exact());
}

TEST_CASE(
    "CheckedCount: sums and joins", "[broad_phase][budget][checked_count]")
{
    CHECK(CheckedCount(SIZE_MAX_) + CheckedCount(0) == CheckedCount(SIZE_MAX_));
    CHECK((CheckedCount(SIZE_MAX_) + CheckedCount(0)).exact());
    const CheckedCount over = CheckedCount(SIZE_MAX_) + CheckedCount(1);
    CHECK(over.overflowed);
    CHECK(over.value == SIZE_MAX_);
    CHECK((CheckedCount(SIZE_MAX_ - 1) + CheckedCount(1)).exact());
    CHECK((CheckedCount(SIZE_MAX_ - 1) + CheckedCount(2)).overflowed);
    // Sticky: adding zero to an overflowed count keeps the flag and SIZE_MAX.
    CHECK((over + CheckedCount(0)) == over);
    CHECK((CheckedCount(0) + over) == over);
    // Join order independence (a parallel reduction joins in any order).
    const CheckedCount a(two_pow(63)), b(two_pow(63)), c(1);
    const CheckedCount abc = (a + b) + c, acb = (a + c) + b, bca = (b + c) + a;
    CHECK(abc == acb);
    CHECK(abc == bca);
    CHECK(abc.overflowed);
    // A representable total is exact in every join order.
    const CheckedCount x(two_pow(63) - two_pow(42)),
        y(two_pow(63) - two_pow(42)), z(7);
    CHECK((x + y) + z == x + (y + z));
    CHECK(((x + y) + z).exact());
    CHECK(((x + y) + z).value == 18446735277616529415ull);
    // operator+= with a plain size_t.
    CheckedCount s;
    s += 5;
    s += CheckedCount(6);
    CHECK(s == CheckedCount(11));
}

TEST_CASE(
    "CheckedCount: unordered pairs n(n-1)/2",
    "[broad_phase][budget][checked_count]")
{
    CHECK(CheckedCount::unordered_pairs(0) == CheckedCount(0));
    CHECK(CheckedCount::unordered_pairs(1) == CheckedCount(0));
    CHECK(CheckedCount::unordered_pairs(2) == CheckedCount(1));
    CHECK(CheckedCount::unordered_pairs(3) == CheckedCount(3));
    CHECK(CheckedCount::unordered_pairs(4) == CheckedCount(6));
    CHECK(CheckedCount::unordered_pairs(5) == CheckedCount(10));
    CHECK(CheckedCount::unordered_pairs(1000) == CheckedCount(499500));
    // n = 2^32 + 1: n(n-1) = 2^64 + 2^32 overflows, n(n-1)/2 = 2^63 + 2^31
    // fits.
    const CheckedCount fits = CheckedCount::unordered_pairs(two_pow(32) + 1);
    CHECK(fits.exact());
    CHECK(fits.value == 9223372039002259456ull);
    CHECK(CheckedCount::product(two_pow(32) + 1, two_pow(32)).overflowed);
    // The largest n whose pair count fits (computed with big integers).
    const size_t n_max = 6074001000ull;
    const CheckedCount largest = CheckedCount::unordered_pairs(n_max);
    CHECK(largest.exact());
    CHECK(largest.value == 18446744070963499500ull);
    CHECK(CheckedCount::unordered_pairs(n_max + 1).overflowed);
    CHECK(CheckedCount::unordered_pairs(SIZE_MAX_).overflowed);
}

TEST_CASE(
    "CheckedCount: budget decisions", "[broad_phase][budget][checked_count]")
{
    CHECK(!CheckedCount(0).exceeds(0));
    CHECK(CheckedCount(1).exceeds(0));
    CHECK(!CheckedCount(100).exceeds(100));             // exact limit
    CHECK(CheckedCount(101).exceeds(100));              // one over
    CHECK(!CheckedCount(SIZE_MAX_).exceeds(SIZE_MAX_)); // largest exact value
    CheckedCount over(SIZE_MAX_);
    over.overflowed = true;
    CHECK(over.exceeds(SIZE_MAX_)); // saturation alone would admit this
    CHECK(over.exceeds(0));
}

TEST_CASE(
    "Broad-phase exceptions: named, not runtime_error",
    "[broad_phase][budget][checked_count]")
{
    static_assert(
        !std::is_base_of_v<std::runtime_error, BroadPhaseUnrepresentable>);
    static_assert(std::is_base_of_v<std::exception, BroadPhaseUnrepresentable>);
    static_assert(
        !std::is_base_of_v<std::runtime_error, BroadPhaseBudgetExceeded>);
    const BroadPhaseBudgetExceeded exact("HashGrid", "cell_items", 5, 4, "d");
    CHECK(exact.exact);
    CHECK_THAT(
        exact.what(), Catch::Matchers::ContainsSubstring("need 5 cell_items"));
    CHECK_THAT(exact.what(), !Catch::Matchers::ContainsSubstring("at least"));
    const BroadPhaseBudgetExceeded lower(
        "HashGrid", "cell_items", SIZE_MAX_, 4, "d", false);
    CHECK(!lower.exact);
    CHECK_THAT(lower.what(), Catch::Matchers::ContainsSubstring("at least"));
    CHECK_THAT(
        lower.what(), Catch::Matchers::ContainsSubstring("not representable"));
    const BroadPhaseUnrepresentable u("HashGrid", "grid_cells", "too many");
    CHECK(u.method == "HashGrid");
    CHECK(u.quantity == "grid_cells");
    CHECK_THAT(
        u.what(), Catch::Matchers::ContainsSubstring("cannot represent"));
    CHECK_THAT(u.what(), Catch::Matchers::ContainsSubstring("too many"));
}

TEST_CASE(
    "Hash grid: an overflowing cell-item count never admits the build",
    "[broad_phase][budget][checked_count]")
{
    CountProbe probe;
    const Eigen::Array3i largest(1 << 21, 1 << 21, (1 << 21) - 1); // 2^63-2^42
    const size_t one_box = 9223367638808264704ull;

    SECTION("control: 1024^3 grid, one box, exact count, rejected")
    {
        probe.set_up(Eigen::Array3i(1024, 1024, 1024), 1);
        probe.budget.max_cell_items = 100;
        CHECK(probe.count() == CheckedCount(two_pow(30)));
        try {
            probe.check();
            FAIL("accepted");
        } catch (const BroadPhaseBudgetExceeded& e) {
            CHECK(e.requested == two_pow(30));
            CHECK(e.exact);
            CHECK_THAT(
                e.what(), !Catch::Matchers::ContainsSubstring("at least"));
            // The byte estimate is items * sizeof(HashItem): 16 bytes per
            // item on LP64, 8 on LLP64 (two longs).
            CHECK_THAT(
                e.what(),
                Catch::Matchers::ContainsSubstring(
                    std::to_string(two_pow(30) * sizeof(HashItem)) + " bytes"));
        }
    }
    if (!KEY_IS_64_BIT) {
        SKIP("the near-2^63 grids need a 64-bit hash key");
    }
    SECTION("one whole-grid box: exact 2^63 - 2^42; exact limit and one under")
    {
        probe.set_up(largest, 1);
        CHECK(probe.count() == CheckedCount(one_box));
        probe.budget.max_cell_items = one_box; // exact limit: admitted
        CHECK_NOTHROW(probe.check());
        probe.budget.max_cell_items = one_box - 1; // one under: rejected
        try {
            probe.check();
            FAIL("accepted");
        } catch (const BroadPhaseBudgetExceeded& e) {
            CHECK(e.requested == one_box);
            CHECK(e.limit == one_box - 1);
            CHECK(e.exact);
            // The byte estimate itself overflows: reported as a bound.
            CHECK_THAT(
                e.what(),
                Catch::Matchers::ContainsSubstring(
                    "more than 18446744073709551615 bytes"));
        }
    }
    SECTION("two whole-grid boxes: exact 2^64 - 2^43 (largest sums)")
    {
        probe.set_up(largest, 2);
        const CheckedCount count = probe.count();
        CHECK(count.exact());
        CHECK(count.value == 18446735277616529408ull);
        probe.budget.max_cell_items = SIZE_MAX_;
        CHECK_NOTHROW(probe.check()); // representable and under SIZE_MAX
        probe.budget.max_cell_items = 100;
        try {
            probe.check();
            FAIL("accepted");
        } catch (const BroadPhaseBudgetExceeded& e) {
            CHECK(e.requested == 18446735277616529408ull);
            CHECK(e.exact);
        }
    }
    SECTION("three whole-grid boxes: the sum overflows (2.77e19)")
    {
        probe.set_up(largest, 3);
        const CheckedCount count = probe.count();
        CHECK(count.overflowed);
        CHECK(count.value == SIZE_MAX_);
        probe.budget.max_cell_items = 100;
        try {
            probe.check();
            FAIL("accepted");
        } catch (const BroadPhaseBudgetExceeded& e) {
            CHECK(!e.exact);
            CHECK(e.requested == SIZE_MAX_);
            CHECK(e.limit == 100);
            CHECK_THAT(
                e.what(),
                Catch::Matchers::ContainsSubstring(
                    "at least 18446744073709551615 cell_items"));
        }
        // A SIZE_MAX budget: saturation alone would admit it, the flag does
        // not.
        probe.budget.max_cell_items = SIZE_MAX_;
        CHECK_THROWS_AS(probe.check(), BroadPhaseBudgetExceeded);
        // Disabled budget: no check, by policy (the count itself is honest).
        probe.budget.max_cell_items = 0;
        CHECK_NOTHROW(probe.check());
        CHECK(probe.count().overflowed);
    }
    SECTION("thread count cannot change the count or the decision")
    {
        // Two whole-grid boxes among many unit boxes: exact, near SIZE_MAX.
        probe.set_up(largest, 2);
        probe.add_unit_boxes(1000);
        const CheckedCount expected(18446735277616529408ull + 1000);
        CheckedCount serial, parallel;
        {
            tbb::global_control one(
                tbb::global_control::max_allowed_parallelism, 1);
            serial = probe.count();
        }
        parallel = probe.count();
        CHECK(serial == expected);
        CHECK(parallel == expected);
        // Three whole-grid boxes among many unit boxes: overflowed either way.
        probe.set_up(largest, 3);
        probe.add_unit_boxes(1000);
        {
            tbb::global_control one(
                tbb::global_control::max_allowed_parallelism, 1);
            serial = probe.count();
        }
        parallel = probe.count();
        CHECK(serial.overflowed);
        CHECK(parallel == serial);
        probe.budget.max_cell_items = SIZE_MAX_;
        CHECK_THROWS_AS(probe.check(), BroadPhaseBudgetExceeded);
        {
            tbb::global_control one(
                tbb::global_control::max_allowed_parallelism, 1);
            CHECK_THROWS_AS(probe.check(), BroadPhaseBudgetExceeded);
        }
    }
    SECTION("zero and one boxes")
    {
        probe.set_up(largest, 0);
        CHECK(probe.count() == CheckedCount(0));
        probe.budget.max_cell_items = 1;
        CHECK_NOTHROW(probe.check());
        probe.add_unit_boxes(1);
        CHECK(probe.count() == CheckedCount(1));
        CHECK_NOTHROW(probe.check()); // exactly at the limit
        probe.add_unit_boxes(1);
        CHECK_THROWS_AS(probe.check(), BroadPhaseBudgetExceeded);
    }
}

TEST_CASE(
    "Hash grid: emission counts use checked pair arithmetic",
    "[broad_phase][budget][checked_count]")
{
    CountProbe probe;
    // Single set: keys 7 (4 items) and 9 (3 items): 6 + 3 = 9 pairs.
    probe.set_items({ { 7, 4 }, { 9, 3 } }, { });
    probe.budget.max_candidate_emissions = 9;
    std::vector<VertexVertexCandidate> vv;
    CHECK_NOTHROW(probe.detect_vertex_vertex_candidates(vv));
    CHECK(vv.empty()); // disjoint boxes
    CHECK(probe.build_statistics().candidate_emissions == 9);
    CHECK(!probe.build_statistics().candidate_emissions_overflowed);
    probe.budget.max_candidate_emissions = 8;
    try {
        probe.detect_vertex_vertex_candidates(vv);
        FAIL("accepted");
    } catch (const BroadPhaseBudgetExceeded& e) {
        CHECK(e.quantity == "candidate_emissions");
        CHECK(e.requested == 9);
        CHECK(e.exact);
    }
    // Two sets: key 7: 4 vertex x 2 edge items = 8; key 9: 3 x 0; key 11: 0
    // x 5.
    probe.set_items({ { 7, 4 }, { 9, 3 } }, { { 7, 2 }, { 11, 5 } });
    probe.budget.max_candidate_emissions = 8;
    std::vector<EdgeVertexCandidate> ev;
    CHECK_NOTHROW(probe.detect_edge_vertex_candidates(ev));
    CHECK(ev.empty());
    CHECK(probe.build_statistics().candidate_emissions == 8);
    probe.budget.max_candidate_emissions = 7;
    CHECK_THROWS_AS(
        probe.detect_edge_vertex_candidates(ev), BroadPhaseBudgetExceeded);
    // Cumulative statistics saturate instead of wrapping.
    BroadPhaseBuildStatistics stats;
    stats.add_candidate_emissions(CheckedCount(SIZE_MAX_ - 1));
    CHECK(stats.candidate_emissions == SIZE_MAX_ - 1);
    CHECK(!stats.candidate_emissions_overflowed);
    stats.add_candidate_emissions(CheckedCount(2));
    CHECK(stats.candidate_emissions == SIZE_MAX_);
    CHECK(stats.candidate_emissions_overflowed);
    CHECK(stats.candidate_emission_count().overflowed);
    stats.add_candidate_emissions(CheckedCount(0));
    CHECK(stats.candidate_emissions_overflowed); // sticky
}

TEST_CASE(
    "Hash grid: grid geometry is validated before any conversion",
    "[broad_phase][budget][checked_count]")
{
    CountProbe probe;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const Eigen::Array3d zero = Eigen::Array3d::Zero();
    const Eigen::Array3d one = Eigen::Array3d::Ones();
    // Invalid cell sizes.
    CHECK_THROWS_AS(probe.resize_public(zero, one, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(
        probe.resize_public(zero, one, -1.0), std::invalid_argument);
    CHECK_THROWS_AS(probe.resize_public(zero, one, nan), std::invalid_argument);
    CHECK_THROWS_AS(probe.resize_public(zero, one, inf), std::invalid_argument);
    // Non-finite extents.
    CHECK_THROWS_AS(
        probe.resize_public(zero, Eigen::Array3d(1, nan, 1), 1.0),
        std::invalid_argument);
    CHECK_THROWS_AS(
        probe.resize_public(zero, Eigen::Array3d(1, inf, 1), 1.0),
        std::invalid_argument);
    CHECK_THROWS_AS(
        probe.resize_public(Eigen::Array3d(-inf, 0, 0), one, 1.0),
        std::invalid_argument);
    // Finite but more cells along an axis than an int holds.
    try {
        probe.resize_public(zero, Eigen::Array3d(3e9, 1, 1), 1.0);
        FAIL("accepted");
    } catch (const BroadPhaseUnrepresentable& e) {
        CHECK(e.method == "HashGrid");
        CHECK(e.quantity == "grid_cells");
        CHECK_THAT(
            e.what(), Catch::Matchers::ContainsSubstring("along an axis"));
    }
    // A division that overflows to infinity is the same failure.
    CHECK_THROWS_AS(
        probe.resize_public(zero, Eigen::Array3d(1e308, 1, 1), 1e-308),
        BroadPhaseUnrepresentable);
    // Per-axis representable, but 2^64 cells in total (the review's grid):
    // no hash key can index them.
    try {
        probe.resize_public(
            zero, Eigen::Array3d(1 << 22, 1 << 21, 1 << 21), 1.0);
        FAIL("accepted");
    } catch (const BroadPhaseUnrepresentable& e) {
        CHECK_THAT(e.what(), Catch::Matchers::ContainsSubstring("hash key"));
    }
    // One more cell than the largest key: refused; fewer: representable,
    // and hash() reaches the last cell.
    if (KEY_IS_64_BIT) {
        CHECK_THROWS_AS(
            probe.resize_public(
                zero, Eigen::Array3d(1 << 21, 1 << 21, 1 << 21), 1.0),
            BroadPhaseUnrepresentable); // 2^63
        CHECK_NOTHROW(probe.resize_public(
            zero, Eigen::Array3d(1 << 21, 1 << 21, (1 << 21) - 1), 1.0));
        CHECK(probe.grid_size()[0] == (1 << 21));
        CHECK(probe.grid_size()[2] == (1 << 21) - 1);
    } else {
        CHECK_THROWS_AS(
            probe.resize_public(zero, Eigen::Array3d(1 << 16, 1 << 15, 1), 1.0),
            BroadPhaseUnrepresentable); // 2^31
        CHECK_NOTHROW(probe.resize_public(
            zero, Eigen::Array3d(1 << 16, (1 << 15) - 1, 1), 1.0));
    }
    // Ordinary grids are unchanged (ceil, at least one cell per axis).
    CHECK_NOTHROW(probe.resize_public(zero, Eigen::Array3d(10, 2.5, 0), 1.0));
    CHECK(probe.grid_size()[0] == 10);
    CHECK(probe.grid_size()[1] == 3);
    CHECK(probe.grid_size()[2] == 1);
    // A box outside the domain, or not finite, never reaches the int cast.
    Eigen::Array3i lo, hi;
    AABB inside;
    inside.min = Eigen::Array3d(0.5, 0.5, 0);
    inside.max = Eigen::Array3d(9.9, 2.4, 0);
    CHECK_NOTHROW(probe.range_public(inside, lo, hi));
    CHECK(lo.x() == 0);
    CHECK(hi.x() == 9);
    CHECK(hi.y() == 2);
    AABB outside = inside;
    outside.max.x() = 50;
    CHECK_THROWS_AS(probe.range_public(outside, lo, hi), std::invalid_argument);
    AABB not_finite = inside;
    not_finite.min.y() = nan;
    CHECK_THROWS_AS(
        probe.range_public(not_finite, lo, hi), std::invalid_argument);
}

TEST_CASE(
    "Hash grid: non-finite vertex positions are refused by name",
    "[broad_phase][budget][checked_count]")
{
    Eigen::MatrixXd V(4, 3);
    V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Eigen::MatrixXi E(3, 2);
    E << 0, 1, 1, 2, 2, 3;
    Eigen::MatrixXi F;
    HashGrid grid;
    CHECK_NOTHROW(grid.build(V, E, F, 1e-3));
    for (const double bad : { std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity() }) {
        for (int row = 0; row < 4; ++row) { // also the first (unmasked) box
            Eigen::MatrixXd Vbad = V;
            Vbad(row, 1) = bad;
            CHECK_THROWS_AS(
                grid.build(Vbad, E, F, 1e-3), std::invalid_argument);
            // A sweep to a non-finite position on the dynamic build.
            CHECK_THROWS_AS(
                grid.build(V, Vbad, E, F, 1e-3), std::invalid_argument);
        }
    }
    // The grid is usable afterwards.
    CHECK_NOTHROW(grid.build(V, E, F, 1e-3));
    std::vector<EdgeVertexCandidate> ev;
    grid.detect_edge_vertex_candidates(ev);
    // Edges 1-2 and 2-3 both reach vertex 0's box; edge 0-1 reaches none.
    CHECK(ev.size() == 2);
}

TEST_CASE(
    "Hash grid: an unrepresentable sweep is refused by name through the public build",
    "[broad_phase][budget][checked_count]")
{
    // Many unit edges (the median box, hence the cell size, stays ~1) and
    // one vertex swept 3e9 units along x: the grid would need 3e9 cells
    // along x, more than an int holds.
    const int n = 40;
    Eigen::MatrixXd V0(n + 1, 3);
    for (int i = 0; i <= n; ++i) {
        V0.row(i) << double(i), 0, 0;
    }
    Eigen::MatrixXi E(n, 2);
    for (int i = 0; i < n; ++i) {
        E.row(i) << i, i + 1;
    }
    Eigen::MatrixXi F;
    Eigen::MatrixXd V1 = V0;
    V1(n, 0) += 3e9;
    const CollisionMesh mesh(V0, E, F);

    for (const bool budgeted : { false, true }) {
        HashGrid grid;
        if (budgeted) {
            grid.budget.max_cell_items = 100;
        }
        Candidates candidates;
        try {
            candidates.build(mesh, V0, V1, 1e-3, &grid);
            FAIL("accepted");
        } catch (const BroadPhaseUnrepresentable& e) {
            CHECK(e.method == "HashGrid");
            CHECK_THAT(
                e.what(), Catch::Matchers::ContainsSubstring("along an axis"));
        }
        CHECK(candidates.empty());
        CHECK(!grid.build_statistics().measured);
        // A later ordinary build on the same objects works (the static
        // build needs 120 items: the budget is raised for it).
        if (budgeted) {
            grid.budget.max_cell_items = 10000;
        }
        CHECK_NOTHROW(candidates.build(mesh, V0, V0, 1e-3, &grid));
        CHECK(grid.build_statistics().vertex_boxes == size_t(n + 1));
        CHECK(grid.build_statistics().cell_items == 120);
    }
    // A moderate sweep of the same vertex is an ordinary (budgeted) build.
    Eigen::MatrixXd V2 = V0;
    V2(n, 0) += 30;
    HashGrid grid;
    grid.budget.max_cell_items = 1000000;
    Candidates candidates;
    CHECK_NOTHROW(candidates.build(mesh, V0, V2, 1e-3, &grid));
    CHECK(grid.build_statistics().measured);
}

TEST_CASE(
    "Brute force: checked pair counts on ordinary sizes",
    "[broad_phase][budget][checked_count]")
{
    // 3 edges = 3 unordered pairs; 4 vertices x 3 edges = 12 rectangular.
    Eigen::MatrixXd V(4, 3);
    V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Eigen::MatrixXi E(3, 2);
    E << 0, 1, 1, 2, 2, 3;
    Eigen::MatrixXi F;
    BruteForce bf;
    bf.budget.max_candidate_emissions = 12;
    bf.build(V, E, F, 1e-3);
    std::vector<EdgeEdgeCandidate> ee;
    std::vector<EdgeVertexCandidate> ev;
    CHECK_NOTHROW(bf.detect_edge_edge_candidates(ee));
    CHECK(bf.build_statistics().candidate_emissions == 3);
    CHECK_NOTHROW(bf.detect_edge_vertex_candidates(ev));
    CHECK(bf.build_statistics().candidate_emissions == 15);
    CHECK(!bf.build_statistics().candidate_emissions_overflowed);
    bf.budget.max_candidate_emissions = 11;
    try {
        bf.detect_edge_vertex_candidates(ev);
        FAIL("accepted");
    } catch (const BroadPhaseBudgetExceeded& e) {
        CHECK(e.requested == 12);
        CHECK(e.exact);
    }
}
