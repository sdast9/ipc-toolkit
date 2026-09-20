#include "hash_grid.hpp"

#include <ipc/broad_phase/voxel_size_heuristic.hpp>
#include <ipc/utils/logger.hpp>
#include <ipc/utils/merge_thread_local.hpp>

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>

#include <algorithm> // std::min/max
#include <cmath>
#include <limits>
#include <stdexcept>

#define IPC_TOOLKIT_HASH_GRID_USE_SORT_UNIQUE // else use unordered_set

using namespace std::placeholders;

namespace ipc {

void HashGrid::build(
    Eigen::ConstRef<Eigen::MatrixXi> edges,
    Eigen::ConstRef<Eigen::MatrixXi> faces)
{
    BroadPhase::build(edges, faces);

    Eigen::Array3d mesh_min, mesh_max;
    compute_mesh_aabb(mesh_min, mesh_max);

    const double cell_size =
        suggest_good_voxel_size(edges.rows() > 0 ? edge_boxes : vertex_boxes);
    resize(mesh_min, mesh_max, cell_size);

    m_build_statistics.cell_size = cell_size;
    for (int i = 0; i < 3; ++i) {
        m_build_statistics.grid_size[i] = grid_size()[i];
    }

    // Opt-in: the exact item count is known from the boxes alone, so the
    // budget is enforced before the first item is allocated.
    check_cell_item_budget();

    insert_boxes();

    m_build_statistics.measured = true;
    m_build_statistics.cell_items =
        vertex_items.size() + edge_items.size() + face_items.size();
}

void HashGrid::box_cell_range(
    const AABB& aabb, Eigen::Array3i& int_min, Eigen::Array3i& int_max) const
{
    const Eigen::Array3d lo = (aabb.min - domain_min()) / cell_size();
    const Eigen::Array3d hi = (aabb.max - domain_min()) / cell_size();

    // A box built from the same vertex boxes as the domain lies in
    // [0, grid_size] (we can round down to -1, but not less, and up to the
    // far border, but not beyond). Anything else -- in particular a
    // non-finite box, which fails every comparison -- must not reach the
    // conversion to int below, whose result would be undefined.
    const Eigen::Array3d upper = grid_size().cast<double>();
    if (!((lo >= -1.0).all() && (lo <= upper).all() && (hi >= -1.0).all()
          && (hi <= upper).all())) {
        throw std::invalid_argument(
            fmt::format(
                "{} broad phase: box [{:g} {:g} {:g}]-[{:g} {:g} {:g}] is not finite or lies outside the grid domain [{:g} {:g} {:g}]-[{:g} {:g} {:g}]",
                name(), aabb.min.x(), aabb.min.y(), aabb.min.z(), aabb.max.x(),
                aabb.max.y(), aabb.max.z(), domain_min().x(), domain_min().y(),
                domain_min().z(), domain_max().x(), domain_max().y(),
                domain_max().z()));
    }

    int_min = lo.cast<int>().max(0).min(grid_size() - 1);
    int_max = hi.cast<int>().max(0).min(grid_size() - 1);
    assert((int_min <= int_max).all());
}

CheckedCount HashGrid::count_cell_items(const AABBs& boxes) const
{
    // Saturating addition with a sticky overflow flag is associative and
    // commutative, so the reduction's result does not depend on how TBB
    // partitions the range or on the number of threads.
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, boxes.size()), CheckedCount(),
        [&](const tbb::blocked_range<size_t>& r, CheckedCount total) {
            Eigen::Array3i int_min, int_max;
            for (size_t i = r.begin(); i != r.end(); ++i) {
                box_cell_range(boxes[i], int_min, int_max);
                // Each extent is at most INT_MAX cells (resize), so these
                // products fit; the sum over the boxes need not.
                total += CheckedCount::product(
                    size_t(int_max.x() - int_min.x() + 1),
                    size_t(int_max.y() - int_min.y() + 1),
                    dim == 3 ? size_t(int_max.z() - int_min.z() + 1) : 1);
            }
            return total;
        },
        [](const CheckedCount& a, const CheckedCount& b) { return a + b; });
}

void HashGrid::check_cell_item_budget() const
{
    if (budget.max_cell_items == 0) {
        return;
    }
    const CheckedCount items = count_cell_items(vertex_boxes)
        + count_cell_items(edge_boxes) + count_cell_items(face_boxes);
    if (items.exceeds(budget.max_cell_items)) {
        const CheckedCount bytes = items.times(sizeof(HashItem));
        throw BroadPhaseBudgetExceeded(
            name(), "cell_items", items.value, budget.max_cell_items,
            fmt::format(
                "{} vertex, {} edge and {} face boxes over a {}x{}x{} grid of cell size {:g} ({}{} bytes of items)",
                vertex_boxes.size(), edge_boxes.size(), face_boxes.size(),
                grid_size()[0], grid_size()[1], grid_size()[2], cell_size(),
                bytes.exact() ? "" : "more than ", bytes.value),
            items.exact());
    }
}

void HashGrid::check_emission_budget(const CheckedCount& emissions) const
{
    m_build_statistics.add_candidate_emissions(emissions);
    if (budget.max_candidate_emissions > 0
        && emissions.exceeds(budget.max_candidate_emissions)) {
        throw BroadPhaseBudgetExceeded(
            name(), "candidate_emissions", emissions.value,
            budget.max_candidate_emissions,
            fmt::format(
                "item pairs sharing a cell in one detect call before filtering and deduplication; {} vertex, {} edge and {} face items over a {}x{}x{} grid of cell size {:g}",
                vertex_items.size(), edge_items.size(), face_items.size(),
                grid_size()[0], grid_size()[1], grid_size()[2], cell_size()),
            emissions.exact());
    }
}

void HashGrid::resize(
    Eigen::ConstRef<Eigen::Array3d> domain_min,
    Eigen::ConstRef<Eigen::Array3d> domain_max,
    const double cell_size)
{
    // Everything below is validated on the floating-point values before
    // any conversion to an integer: a conversion of a non-finite or
    // out-of-range value is undefined behavior, and release builds have no
    // assertions to catch it.
    if (!(cell_size > 0.0) || !std::isfinite(cell_size)) {
        throw std::invalid_argument(
            fmt::format(
                "{} broad phase: cell size {:g} is not a positive finite number",
                name(), cell_size));
    }
    const Eigen::Array3d extent = domain_max - domain_min;
    if (!extent.isFinite().all() || (extent < 0.0).any()) {
        throw std::invalid_argument(
            fmt::format(
                "{} broad phase: the mesh bounding box [{:g} {:g} {:g}]-[{:g} {:g} {:g}] is not finite",
                name(), domain_min.x(), domain_min.y(), domain_min.z(),
                domain_max.x(), domain_max.y(), domain_max.z()));
    }

    // Cells per axis as doubles (the division can overflow to infinity),
    // then their product in checked integer arithmetic against the key
    // type, which hash() indexes the cells with.
    const Eigen::Array3d cells = (extent / cell_size).ceil().max(1.0);
    const double max_axis = double(std::numeric_limits<int>::max());
    if (!cells.isFinite().all() || (cells > max_axis).any()) {
        throw BroadPhaseUnrepresentable(
            name(), "grid_cells",
            fmt::format(
                "{:g} x {:g} x {:g} cells of size {:g} over an extent of {:g} x {:g} x {:g}: more than {} cells along an axis",
                cells.x(), cells.y(), cells.z(), cell_size, extent.x(),
                extent.y(), extent.z(), std::numeric_limits<int>::max()));
    }
    const CheckedCount total = CheckedCount::product(
        size_t(cells.x()), size_t(cells.y()), size_t(cells.z()));
    if (total.exceeds(size_t(std::numeric_limits<long>::max()))) {
        throw BroadPhaseUnrepresentable(
            name(), "grid_cells",
            fmt::format(
                "{:g} x {:g} x {:g} cells of size {:g} over an extent of {:g} x {:g} x {:g}: more than the {} cells a hash key can index on this platform",
                cells.x(), cells.y(), cells.z(), cell_size, extent.x(),
                extent.y(), extent.z(), std::numeric_limits<long>::max()));
    }

    m_domain_min = domain_min;
    m_domain_max = domain_max;
    m_cell_size = cell_size;
    m_grid_size = cells.cast<int>();

    logger().trace(
        "hash-grid resized with a size of {:d}x{:d}x{:d}", grid_size()[0],
        grid_size()[1], grid_size()[2]);
}

void HashGrid::insert_boxes()
{
    insert_boxes(this->vertex_boxes, vertex_items);
    insert_boxes(this->edge_boxes, edge_items);
    insert_boxes(this->face_boxes, face_items);
}

void HashGrid::insert_boxes(
    const AABBs& boxes, std::vector<HashItem>& items) const
{
    tbb::enumerable_thread_specific<std::vector<HashItem>> storage;

    tbb::parallel_for(0L, long(boxes.size()), [&](long i) {
        insert_box(boxes[i], i, storage.local());
    });

    merge_thread_local_vectors(storage, items);

    // Sorted all they (key, value) pairs, where key is the hash key, and
    // value is the element index
    tbb::parallel_sort(items.begin(), items.end());
}

void HashGrid::insert_box(
    const AABB& aabb, const long id, std::vector<HashItem>& items) const
{
    Eigen::Array3i int_min, int_max;
    box_cell_range(aabb, int_min, int_max);

    const int min_z = dim == 3 ? int_min.z() : 0;
    const int max_z = dim == 3 ? int_max.z() : 0;
    for (int x = int_min.x(); x <= int_max.x(); ++x) {
        for (int y = int_min.y(); y <= int_max.y(); ++y) {
            for (int z = min_z; z <= max_z; ++z) {
                items.emplace_back(hash(x, y, z), id);
            }
        }
    }
}

template <typename Candidate>
void HashGrid::detect_candidates(
    const std::vector<HashItem>& items0,
    const std::vector<HashItem>& items1,
    const AABBs& boxes0,
    const AABBs& boxes1,
    const std::function<bool(size_t, size_t)>& can_collide,
    std::vector<Candidate>& candidates) const
{
    // Entries with the same key means they share a cell (that cell index
    // hashes to the same key) and should be flagged for low-level intersection
    // testing. We loop over the entire sorted set of (key,value) pairs
    // creating Candidate entries for pairs with the same key

    // 1. Soft merge of items (assuming items are sorted)
    size_t num_items = items0.size() + items1.size();
    std::vector<long> merged_item_indices;
    merged_item_indices.reserve(num_items);
    {
        long i = 0, j = 0;
        while (i < items0.size() && j < items1.size()) {
            if (items0[i] < items1[j]) {
                merged_item_indices.push_back(-(i++) - 1);
            } else {
                merged_item_indices.push_back(j++);
            }
        }
        while (i < items0.size()) {
            merged_item_indices.push_back(-(i++) - 1);
        }
        while (j < items1.size()) {
            merged_item_indices.push_back(j++);
        }
    }
    assert(merged_item_indices.size() == num_items);

    const auto get_item = [&](long i) -> const HashItem& {
        return i < 0 ? items0[-(i + 1)] : items1[i];
    };

    // Opt-in: the exact number of (item0, item1) pairs sharing a key is
    // known from the sorted items, before any candidate is allocated.
    if (budget.enabled()) {
        CheckedCount emissions;
        for (size_t i = 0; i < num_items;) {
            const long key = get_item(merged_item_indices[i]).key;
            size_t n0 = 0, n1 = 0;
            for (; i < num_items && get_item(merged_item_indices[i]).key == key;
                 ++i) {
                (merged_item_indices[i] < 0 ? n0 : n1)++;
            }
            emissions += CheckedCount::product(n0, n1);
        }
        check_emission_budget(emissions);
    }

    // 2. Enumerate hash collisions
#ifdef IPC_TOOLKIT_HASH_GRID_USE_SORT_UNIQUE
    tbb::enumerable_thread_specific<std::vector<Candidate>> storage;
#else
    tbb::enumerable_thread_specific<unordered_set<Candidate>> storage;
#endif

    tbb::parallel_for(
        tbb::blocked_range2d<long>(0L, num_items - 1, 0L, num_items),
        [&](const tbb::blocked_range2d<long>& r) {
            auto& local_candidates = storage.local();

            // i < j
            long i_end = std::min(r.rows().end(), r.cols().end());
            for (long i = r.rows().begin(); i < i_end; i++) {
                const long idx0 = merged_item_indices[i];
                const HashItem& item0 = get_item(idx0);

                // i < r.cols().end() → i + 1 <= r.cols().end()
                long j_begin = std::max(i + 1, r.cols().begin());
                for (long j = j_begin; j < r.cols().end(); j++) {
                    const long idx1 = merged_item_indices[j];
                    const HashItem& item1 = get_item(idx1);

                    if (item0.key != item1.key) {
                        break; // This avoids a brute force comparison
                    }

                    long id0 = item0.id, id1 = item1.id;
                    if (idx0 >= 0 && idx1 < 0) {
                        std::swap(id0, id1);
                    } else if (idx0 >= 0 || idx1 < 0) {
                        continue;
                    }
                    assert(id0 < boxes0.size() && id1 < boxes1.size());

                    if (!can_collide(id0, id1)) {
                        continue;
                    }

                    if (boxes0[id0].intersects(boxes1[id1])) {
#ifdef IPC_TOOLKIT_HASH_GRID_USE_SORT_UNIQUE
                        local_candidates.emplace_back(id0, id1);
#else
                        local_candidates.emplace(id0, id1);
#endif
                    }
                }
            }
        });

#ifdef IPC_TOOLKIT_HASH_GRID_USE_SORT_UNIQUE
    merge_thread_local_vectors(storage, candidates);

    // Remove the duplicate candidates
    tbb::parallel_sort(candidates.begin(), candidates.end());
    auto new_end = std::unique(candidates.begin(), candidates.end());
    candidates.erase(new_end, candidates.end());
#else
    unordered_set<Candidate> candidates_set;
    merge_thread_local_unordered_sets(storage, candidates_set);

    candidates.reserve(candidates_set.size());
    candidates.insert(
        candidates.end(), candidates_set.begin(), candidates_set.end());
#endif
}

template <typename Candidate>
void HashGrid::detect_candidates(
    const std::vector<HashItem>& items,
    const AABBs& boxes,
    const std::function<bool(size_t, size_t)>& can_collide,
    std::vector<Candidate>& candidates) const
{
    // Entries with the same key means they share a cell (that cell index
    // hashes to the same key) and should be flagged for low-level
    // intersection testing. So we loop over the entire sorted set of
    // (key,value) pairs creating Candidate entries for pairs with the same key

    // Opt-in: the exact number of item pairs sharing a key is known from
    // the sorted items, before any candidate is allocated.
    if (budget.enabled()) {
        CheckedCount emissions;
        for (size_t i = 0; i < items.size();) {
            size_t j = i;
            while (j < items.size() && items[j].key == items[i].key) {
                ++j;
            }
            emissions += CheckedCount::unordered_pairs(j - i);
            i = j;
        }
        check_emission_budget(emissions);
    }

#ifdef IPC_TOOLKIT_HASH_GRID_USE_SORT_UNIQUE
    tbb::enumerable_thread_specific<std::vector<Candidate>> storage;
#else
    tbb::enumerable_thread_specific<unordered_set<Candidate>> storage;
#endif

    tbb::parallel_for(
        tbb::blocked_range2d<long>(0L, items.size() - 1, 0L, items.size()),
        [&](const tbb::blocked_range2d<long>& r) {
            auto& local_candidates = storage.local();

            // i < j
            long i_end = std::min(r.rows().end(), r.cols().end());
            for (long i = r.rows().begin(); i < i_end; i++) {
                const HashItem& item0 = items[i];
                const AABB& box0 = boxes[item0.id];

                // i < r.cols().end() → i + 1 <= r.cols().end()
                long j_begin = std::max(i + 1, r.cols().begin());
                assert(j_begin > i);
                for (long j = j_begin; j < r.cols().end(); j++) {
                    const HashItem& item1 = items[j];

                    if (item0.key != item1.key) {
                        break; // This avoids a brute force comparison
                    }

                    if (!can_collide(item0.id, item1.id)) {
                        continue;
                    }

                    const AABB& box1 = boxes[item1.id];
                    if (box0.intersects(box1)) {
#ifdef IPC_TOOLKIT_HASH_GRID_USE_SORT_UNIQUE
                        local_candidates.emplace_back(item0.id, item1.id);
#else
                        local_candidates.emplace(item0.id, item1.id);
#endif
                    }
                }
            }
        });

#ifdef IPC_TOOLKIT_HASH_GRID_USE_SORT_UNIQUE
    merge_thread_local_vectors(storage, candidates);

    // Remove the duplicate candidates
    tbb::parallel_sort(candidates.begin(), candidates.end());
    auto new_end = std::unique(candidates.begin(), candidates.end());
    candidates.erase(new_end, candidates.end());
#else
    unordered_set<Candidate> candidates_set;
    merge_thread_local_unordered_sets(storage, candidates_set);

    candidates.reserve(candidates_set.size());
    candidates.insert(
        candidates.end(), candidates_set.begin(), candidates_set.end());
#endif
}

void HashGrid::detect_vertex_vertex_candidates(
    std::vector<VertexVertexCandidate>& candidates) const
{
    detect_candidates(
        vertex_items, vertex_boxes, can_vertices_collide, candidates);
}

void HashGrid::detect_edge_vertex_candidates(
    std::vector<EdgeVertexCandidate>& candidates) const
{
    detect_candidates(
        edge_items, vertex_items, edge_boxes, vertex_boxes,
        std::bind(&HashGrid::can_edge_vertex_collide, this, _1, _2),
        candidates);
}

void HashGrid::detect_edge_edge_candidates(
    std::vector<EdgeEdgeCandidate>& candidates) const
{
    detect_candidates(
        edge_items, edge_boxes,
        std::bind(&HashGrid::can_edges_collide, this, _1, _2), candidates);
}

void HashGrid::detect_face_vertex_candidates(
    std::vector<FaceVertexCandidate>& candidates) const
{
    detect_candidates(
        face_items, vertex_items, face_boxes, vertex_boxes,
        std::bind(&HashGrid::can_face_vertex_collide, this, _1, _2),
        candidates);
}

void HashGrid::detect_edge_face_candidates(
    std::vector<EdgeFaceCandidate>& candidates) const
{
    detect_candidates(
        edge_items, face_items, edge_boxes, face_boxes,
        std::bind(&HashGrid::can_edge_face_collide, this, _1, _2), candidates);
}

void HashGrid::detect_face_face_candidates(
    std::vector<FaceFaceCandidate>& candidates) const
{
    detect_candidates(
        face_items, face_boxes,
        std::bind(&HashGrid::can_faces_collide, this, _1, _2), candidates);
}

} // namespace ipc
