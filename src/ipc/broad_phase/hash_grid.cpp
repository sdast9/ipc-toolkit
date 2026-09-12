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
    int_min = ((aabb.min - domain_min()) / cell_size()).cast<int>();
    // We can round down to -1, but not less
    assert((int_min >= -1).all());
    assert((int_min <= grid_size()).all());
    int_min = int_min.max(0).min(grid_size() - 1);

    int_max = ((aabb.max - domain_min()) / cell_size()).cast<int>();
    assert((int_max >= -1).all());
    assert((int_max <= grid_size()).all());
    int_max = int_max.max(0).min(grid_size() - 1);
    assert((int_min <= int_max).all());
}

size_t HashGrid::count_cell_items(const AABBs& boxes) const
{
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, boxes.size()), size_t(0),
        [&](const tbb::blocked_range<size_t>& r, size_t total) {
            Eigen::Array3i int_min, int_max;
            for (size_t i = r.begin(); i != r.end(); ++i) {
                box_cell_range(boxes[i], int_min, int_max);
                size_t cells = size_t(int_max.x() - int_min.x() + 1)
                    * size_t(int_max.y() - int_min.y() + 1);
                if (dim == 3) {
                    cells *= size_t(int_max.z() - int_min.z() + 1);
                }
                total += cells;
            }
            return total;
        },
        std::plus<size_t>());
}

void HashGrid::check_cell_item_budget() const
{
    if (budget.max_cell_items == 0) {
        return;
    }
    const size_t items = count_cell_items(vertex_boxes)
        + count_cell_items(edge_boxes) + count_cell_items(face_boxes);
    if (items > budget.max_cell_items) {
        throw BroadPhaseBudgetExceeded(
            name(), "cell_items", items, budget.max_cell_items,
            fmt::format(
                "{} vertex, {} edge and {} face boxes over a {}x{}x{} grid of cell size {:g} ({} bytes of items)",
                vertex_boxes.size(), edge_boxes.size(), face_boxes.size(),
                grid_size()[0], grid_size()[1], grid_size()[2], cell_size(),
                items * sizeof(HashItem)));
    }
}

void HashGrid::check_emission_budget(const size_t emissions) const
{
    m_build_statistics.candidate_emissions += emissions;
    if (budget.max_candidate_emissions > 0
        && emissions > budget.max_candidate_emissions) {
        throw BroadPhaseBudgetExceeded(
            name(), "candidate_emissions", emissions,
            budget.max_candidate_emissions,
            fmt::format(
                "item pairs sharing a cell in one detect call before filtering and deduplication; {} vertex, {} edge and {} face items over a {}x{}x{} grid of cell size {:g}",
                vertex_items.size(), edge_items.size(), face_items.size(),
                grid_size()[0], grid_size()[1], grid_size()[2], cell_size()));
    }
}

void HashGrid::resize(
    Eigen::ConstRef<Eigen::Array3d> domain_min,
    Eigen::ConstRef<Eigen::Array3d> domain_max,
    const double cell_size)
{
    assert(cell_size > 0.0);
    assert(std::isfinite(cell_size));

    m_domain_min = domain_min;
    m_domain_max = domain_max;
    m_cell_size = cell_size;
    m_grid_size =
        ((domain_max - domain_min) / cell_size).ceil().cast<int>().max(1);

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
        size_t emissions = 0;
        for (size_t i = 0; i < num_items;) {
            const long key = get_item(merged_item_indices[i]).key;
            size_t n0 = 0, n1 = 0;
            for (; i < num_items && get_item(merged_item_indices[i]).key == key;
                 ++i) {
                (merged_item_indices[i] < 0 ? n0 : n1)++;
            }
            emissions += n0 * n1;
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
        size_t emissions = 0;
        for (size_t i = 0; i < items.size();) {
            size_t j = i;
            while (j < items.size() && items[j].key == items[i].key) {
                ++j;
            }
            const size_t n = j - i;
            emissions += n * (n - 1) / 2;
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
