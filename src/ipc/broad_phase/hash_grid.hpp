#pragma once

#include <ipc/broad_phase/broad_phase.hpp>

namespace ipc {

/// @brief An entry into the hash grid as a (key, value) pair.
struct HashItem {
    /// @brief The key of the item (a cell index; HashGrid::resize refuses a
    ///        grid with more cells than this type holds).
    long key;
    /// @brief The value of the item.
    long id;

    /// @brief Construct a hash item as a (key, value) pair.
    HashItem(long _key, long _id) : key(_key), id(_id) { }

    /// @brief Compare HashItems by their keys for sorting.
    bool operator<(const HashItem& other) const
    {
        if (key == other.key) {
            return id < other.id;
        }
        return key < other.key;
    }
};

/// @brief Hash grid broad phase collision detection.
class HashGrid : public BroadPhase {
public:
    HashGrid() = default;

    /// @brief Get the name of the broad phase method.
    /// @return The name of the broad phase method.
    std::string name() const override { return "HashGrid"; }

    using BroadPhase::build;

    /// @brief The hash grid enforces BroadPhaseBudget: cell items are counted
    ///        from the boxes before insertion, pair emissions from the sorted
    ///        items before enumeration.
    bool supports_budget() const override { return true; }

    /// @brief Clear the hash grid.
    void clear() override
    {
        BroadPhase::clear();
        vertex_items.clear();
        edge_items.clear();
        face_items.clear();
    }

    /// @brief Find the candidate vertex-vertex collisions.
    void detect_vertex_vertex_candidates(
        std::vector<VertexVertexCandidate>& candidates) const override;

    /// @brief Find the candidate edge-vertex collisions.
    /// @param[out] candidates The candidate edge-vertex collisions.
    void detect_edge_vertex_candidates(
        std::vector<EdgeVertexCandidate>& candidates) const override;

    /// @brief Find the candidate edge-edge collisions.
    /// @param[out] candidates The candidate edge-edge collisions.
    void detect_edge_edge_candidates(
        std::vector<EdgeEdgeCandidate>& candidates) const override;

    /// @brief Find the candidate face-vertex collisions.
    /// @param[out] candidates The candidate face-vertex collisions.
    void detect_face_vertex_candidates(
        std::vector<FaceVertexCandidate>& candidates) const override;

    /// @brief Find the candidate edge-face intersections.
    /// @param[out] candidates The candidate edge-face intersections.
    void detect_edge_face_candidates(
        std::vector<EdgeFaceCandidate>& candidates) const override;

    /// @brief Find the candidate face-face collisions.
    /// @param[out] candidates The candidate face-face collisions.
    void detect_face_face_candidates(
        std::vector<FaceFaceCandidate>& candidates) const override;

    double cell_size() const { return m_cell_size; }
    // TODO: Update this
    const Eigen::Array3i& grid_size() const { return m_grid_size; }
    const Eigen::Array3d& domain_min() const { return m_domain_min; }
    const Eigen::Array3d& domain_max() const { return m_domain_max; }

protected:
    /// @brief Build the broad phase for collision detection.
    /// @note Assumes the vertex_boxes have been built.
    /// @param edges Collision mesh edges
    /// @param faces Collision mesh faces
    void build(
        Eigen::ConstRef<Eigen::MatrixXi> edges,
        Eigen::ConstRef<Eigen::MatrixXi> faces) override;

    /// @brief Size the grid over [domain_min, domain_max] with cubic cells.
    /// @throws std::invalid_argument if the cell size is not a positive
    ///         finite number or the domain is not finite.
    /// @throws BroadPhaseUnrepresentable if the grid would have more than
    ///         INT_MAX cells along an axis or more cells in total than a
    ///         hash key (long) can index — checked in integer arithmetic
    ///         before any floating-point-to-integer conversion.
    void resize(
        Eigen::ConstRef<Eigen::Array3d> domain_min,
        Eigen::ConstRef<Eigen::Array3d> domain_max,
        double cell_size);

    void insert_boxes();

    void insert_boxes(const AABBs& boxes, std::vector<HashItem>& items) const;

    /// @brief Add an AABB of the extents to the hash grid.
    void insert_box(
        const AABB& aabb, const long id, std::vector<HashItem>& items) const;

    /// @brief Inclusive cell index range an AABB covers (clamped to the grid).
    ///        Shared by insert_box and the pre-insertion item count so the
    ///        count is exactly the number of items insert_box emits.
    /// @throws std::invalid_argument if the box is not finite or lies
    ///         outside the grid's domain (checked on the floating-point
    ///         coordinates, before the conversion to cell indices).
    void box_cell_range(
        const AABB& aabb,
        Eigen::Array3i& int_min,
        Eigen::Array3i& int_max) const;

    /// @brief Number of items insert_box would emit for the boxes (no
    ///        allocation). Checked arithmetic: an overflowing sum is
    ///        reported as such, never wrapped.
    CheckedCount count_cell_items(const AABBs& boxes) const;

    /// @brief Check the cell-item budget from the boxes, before any item is inserted.
    void check_cell_item_budget() const;

    /// @brief Check the emission budget of one detect call from the sorted items.
    /// @param emissions Pre-filter pair count of the enumeration (exact, or
    ///        saturated with its overflow flag set).
    void check_emission_budget(const CheckedCount& emissions) const;

    /// @brief Create the hash of a cell location: its row-major cell index,
    ///        computed in the key type. resize() guarantees the grid's cell
    ///        count fits a long, so this cannot overflow.
    long hash(int x, int y, int z) const
    {
        assert(x >= 0 && y >= 0 && z >= 0);
        assert(
            x < grid_size()[0] && y < grid_size()[1]
            && (grid_size().size() == 2 || z < grid_size()[2]));
        return (long(z) * grid_size()[1] + y) * grid_size()[0] + x;
    }

private:
    /// @brief Find the candidate collisions between two sets of items.
    /// @tparam Candidate The type of collision candidate.
    /// @param[in] items0 First set of items.
    /// @param[in] items1 Second set of items.
    /// @param[in] boxes0 First set's boxes.
    /// @param[in] boxes1 Second set's boxes.
    /// @param[in] can_collide Function to determine if two items can collide.
    /// @param[out] candidates The candidate collisions.
    template <typename Candidate>
    void detect_candidates(
        const std::vector<HashItem>& items0,
        const std::vector<HashItem>& items1,
        const AABBs& boxes0,
        const AABBs& boxes1,
        const std::function<bool(size_t, size_t)>& can_collide,
        std::vector<Candidate>& candidates) const;

    /// @brief Find the candidate collisions among a set of items.
    /// @tparam Candidate The type of collision candidate.
    /// @param[in] items The set of items.
    /// @param[in] boxes The items' boxes.
    /// @param[in] can_collide Function to determine if two items can collide.
    /// @param[out] candidates The candidate collisions.
    template <typename Candidate>
    void detect_candidates(
        const std::vector<HashItem>& items,
        const AABBs& boxes,
        const std::function<bool(size_t, size_t)>& can_collide,
        std::vector<Candidate>& candidates) const;

protected:
    double m_cell_size = -1;
    Eigen::Array3i m_grid_size;
    Eigen::Array3d m_domain_min;
    Eigen::Array3d m_domain_max;

    std::vector<HashItem> vertex_items;
    std::vector<HashItem> edge_items;
    std::vector<HashItem> face_items;
};

} // namespace ipc
