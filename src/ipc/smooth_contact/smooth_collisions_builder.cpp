#include "smooth_collisions_builder.hpp"

#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_triangle.hpp>

#include <tbb/enumerable_thread_specific.h>

#include <algorithm>
#include <cassert>

namespace ipc {

namespace {
    template <int dim, typename TCollision>
    void add_collision(
        const std::shared_ptr<TCollision>& pair,
        unordered_map<std::pair<index_t, index_t>, std::shared_ptr<TCollision>>&
            cc_to_id,
        std::vector<std::shared_ptr<SmoothCollision>>& collisions)
    {
        assert(pair != nullptr);
        if (pair->is_active()
            && cc_to_id.find(pair->get_hash()) == cc_to_id.end()) {
            // New collision, so add it to the end of collisions
            cc_to_id.emplace(pair->get_hash(), pair);
            collisions.push_back(pair);
        }
    }

    template <int dim, typename TCollision>
    void add_collision(
        const std::shared_ptr<TCollision>& pair,
        std::vector<std::shared_ptr<SmoothCollision>>& collisions)
    {
        assert(pair != nullptr);
        if (pair->is_active()) {
            collisions.push_back(pair);
        }
    }

    // -------------------------------------------------------------------------
    // Canonical collision order (see SmoothCollisionsBuilder::merge).

    using CollisionKey = std::pair<index_t, index_t>;

    /// Gather the deduplicated collisions of one kind from every thread's map
    /// and append them in ascending key order (key = the pair of primitive ids
    /// the map is keyed by). Neither the maps' iteration order nor the order of
    /// the thread-local builders may influence the output: the maps use a hash
    /// whose seed changes between processes (Abseil), so emitting in iteration
    /// order made the summation order of the potential, gradient, and Hessian
    /// differ between otherwise identical runs.
    ///
    /// Several threads (or several candidates in one thread) can create the
    /// same key. Every such record is built from the same inputs (the key's
    /// primitive ids, the parameters, the key's dhat and the vertex positions),
    /// so the records are interchangeable and keeping any one of them is
    /// deterministic; the assertion checks that invariant.
    template <typename TCollision, typename Builder, typename Map>
    size_t append_in_key_order(
        const tbb::enumerable_thread_specific<Builder>& local_storage,
        Map Builder::* map,
        std::vector<std::shared_ptr<SmoothCollision>>& collisions)
    {
        std::vector<std::pair<CollisionKey, std::shared_ptr<TCollision>>>
            entries;
        for (const Builder& builder : local_storage) {
            for (const auto& [key, collision] : builder.*map) {
                entries.emplace_back(key, collision);
            }
        }

        std::sort(
            entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        size_t count = 0;
        for (size_t i = 0; i < entries.size(); i++) {
            if (i > 0 && entries[i].first == entries[i - 1].first) {
                assert(
                    entries[i].second->dhat() == entries[i - 1].second->dhat()
                    && entries[i].second->vertex_ids()
                        == entries[i - 1].second->vertex_ids());
                continue; // duplicate key: an interchangeable record
            }
            collisions.push_back(entries[i].second);
            count++;
        }
        return count;
    }

    /// Append the (not deduplicated) collisions of the given type from every
    /// thread's list, sorted by their primitive ids. Records with equal ids are
    /// built from the same inputs, so their relative order does not matter.
    template <typename Builder>
    size_t append_type_in_key_order(
        const tbb::enumerable_thread_specific<Builder>& local_storage,
        const CollisionType type,
        std::vector<std::shared_ptr<SmoothCollision>>& collisions)
    {
        std::vector<std::shared_ptr<SmoothCollision>> of_type;
        for (const Builder& builder : local_storage) {
            for (const auto& cc : builder.collisions) {
                if (cc->type() == type) {
                    of_type.push_back(cc);
                }
            }
        }

        std::sort(
            of_type.begin(), of_type.end(),
            [](const auto& a, const auto& b) {
                return a->get_hash() < b->get_hash();
            });

        collisions.insert(collisions.end(), of_type.begin(), of_type.end());
        return of_type.size();
    }
} // namespace

void SmoothCollisionsBuilder<2>::add_edge_vertex_collisions(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    const std::vector<EdgeVertexCandidate>& candidates,
    const SmoothContactParameters& params,
    const std::function<double(const index_t)>& vert_dhat,
    const std::function<double(const index_t)>& edge_dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [ei, vi] = candidates[i];

        const PointEdgeDistanceType pe_dtype = point_edge_distance_type(
            vertices.row(vi), vertices.row(mesh.edges()(ei, 0)),
            vertices.row(mesh.edges()(ei, 1)));

        if (pe_dtype == PointEdgeDistanceType::P_E) {
            add_collision<2, SmoothCollisionTemplate<Edge2, Point2>>(
                std::make_shared<SmoothCollisionTemplate<Edge2, Point2>>(
                    ei, vi, pe_dtype, mesh, params,
                    std::min(edge_dhat(ei), vert_dhat(vi)), vertices),
                vert_edge_2_to_id, collisions);
        }

        for (int j : { 0, 1 }) {
            const auto& vj = mesh.edges()(ei, j);
            const double dhat = std::min(vert_dhat(vi), vert_dhat(vj));
            if ((vertices.row(vi) - vertices.row(vj)).norm() >= dhat) {
                continue;
            }
            add_collision<2, SmoothCollisionTemplate<Point2, Point2>>(
                std::make_shared<SmoothCollisionTemplate<Point2, Point2>>(
                    std::min<index_t>(vi, vj), std::max<index_t>(vi, vj),
                    PointPointDistanceType::P_P, mesh, params, dhat, vertices),
                vert_vert_2_to_id, collisions);
        }
    }
}

// ============================================================================

void SmoothCollisionsBuilder<3>::add_edge_edge_collisions(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    const std::vector<EdgeEdgeCandidate>& candidates,
    const SmoothContactParameters& params,
    const std::function<double(const index_t)>& vert_dhat,
    const std::function<double(const index_t)>& edge_dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [eai, ebi] = candidates[i];

        const auto [ea0, ea1, eb0, eb1] =
            candidates[i].vertices(vertices, mesh.edges(), mesh.faces());

        const EdgeEdgeDistanceType actual_dtype =
            edge_edge_distance_type(ea0, ea1, eb0, eb1);

        const double distance =
            sqrt(edge_edge_distance(ea0, ea1, eb0, eb1, actual_dtype));

        if (actual_dtype != EdgeEdgeDistanceType::EA_EB
            || distance >= params.dhat) {
            continue;
        }

        add_collision<3, SmoothCollisionTemplate<Edge3, Edge3>>(
            std::make_shared<SmoothCollisionTemplate<Edge3, Edge3>>(
                std::min(eai, ebi), std::max(eai, ebi), actual_dtype, mesh,
                params, std::min(edge_dhat(eai), edge_dhat(ebi)), vertices),
            collisions);
    }
}

void SmoothCollisionsBuilder<3>::add_face_vertex_collisions(
    const CollisionMesh& mesh,
    Eigen::ConstRef<Eigen::MatrixXd> vertices,
    const std::vector<FaceVertexCandidate>& candidates,
    const SmoothContactParameters& params,
    const std::function<double(const index_t)>& vert_dhat,
    const std::function<double(const index_t)>& edge_dhat,
    const std::function<double(const index_t)>& face_dhat,
    const size_t start_i,
    const size_t end_i)
{
    for (size_t i = start_i; i < end_i; i++) {
        const auto& [fi, vi] = candidates[i];

        Eigen::Vector3d v = vertices.row(vi);
        Eigen::Vector3d f0 = vertices.row(mesh.faces()(fi, 0));
        Eigen::Vector3d f1 = vertices.row(mesh.faces()(fi, 1));
        Eigen::Vector3d f2 = vertices.row(mesh.faces()(fi, 2));
        const PointTriangleDistanceType pt_dtype =
            point_triangle_distance_type(v, f0, f1, f2);
        const double distance =
            sqrt(point_triangle_distance(v, f0, f1, f2, pt_dtype));

        if (distance >= vert_dhat(vi)) {
            continue;
        }

        if (pt_dtype == PointTriangleDistanceType::P_T) {
            add_collision<3, SmoothCollisionTemplate<Face, Point3>>(
                std::make_shared<SmoothCollisionTemplate<Face, Point3>>(
                    fi, vi, pt_dtype, mesh, params,
                    std::min(face_dhat(fi), vert_dhat(vi)), vertices),
                collisions);
        }

        for (int lv = 0; lv < 3; lv++) {
            const auto& vj = mesh.faces()(fi, lv);
            const double dhat = std::min(vert_dhat(vi), vert_dhat(vj));
            if ((vertices.row(vi) - vertices.row(vj)).norm() >= dhat) {
                continue;
            }
            add_collision<3, SmoothCollisionTemplate<Point3, Point3>>(
                std::make_shared<SmoothCollisionTemplate<Point3, Point3>>(
                    std::min<index_t>(vi, vj), std::max<index_t>(vi, vj),
                    PointPointDistanceType::P_P, mesh, params, dhat, vertices),
                vert_vert_3_to_id, collisions);
        }

        for (int le = 0; le < 3; le++) {
            const auto& eid = mesh.faces_to_edges()(fi, le);
            const double dhat = std::min(edge_dhat(eid), vert_dhat(vi));

            const PointEdgeDistanceType pe_dtype = point_edge_distance_type(
                vertices.row(vi), vertices.row(mesh.edges()(eid, 0)),
                vertices.row(mesh.edges()(eid, 1)));
            const double distance_sqr = point_edge_distance(
                vertices.row(vi), vertices.row(mesh.edges()(eid, 0)),
                vertices.row(mesh.edges()(eid, 1)), pe_dtype);

            if (pe_dtype != PointEdgeDistanceType::P_E
                || sqrt(distance_sqr) >= dhat) {
                continue;
            }

            add_collision<3, SmoothCollisionTemplate<Edge3, Point3>>(
                std::make_shared<SmoothCollisionTemplate<Edge3, Point3>>(
                    eid, vi, pe_dtype, mesh, params, dhat, vertices),
                edge_vert_3_to_id, collisions);
        }
    }
}

void SmoothCollisionsBuilder<3>::merge(
    const tbb::enumerable_thread_specific<SmoothCollisionsBuilder<3>>&
        local_storage,
    SmoothCollisions& merged_collisions)
{
    // size up the hash items
    size_t total = 0;
    for (const auto& storage : local_storage) {
        total += storage.collisions.size();
    }

    merged_collisions.collisions.reserve(total);

    // Emit a canonical sequence that depends only on the contact set, not on
    // hash iteration or thread scheduling: vertex-vertex, edge-vertex,
    // edge-edge, then face-vertex collisions, each sorted by primitive ids.
    // The collision objects themselves (orientation, weight, dhat) are kept
    // as built.
    auto& collisions = merged_collisions.collisions;
    const size_t vert_vert_count =
        append_in_key_order<SmoothCollisionTemplate<Point3, Point3>>(
            local_storage, &SmoothCollisionsBuilder<3>::vert_vert_3_to_id,
            collisions);
    const size_t edge_vert_count =
        append_in_key_order<SmoothCollisionTemplate<Edge3, Point3>>(
            local_storage, &SmoothCollisionsBuilder<3>::edge_vert_3_to_id,
            collisions);
    const size_t edge_edge_count = append_type_in_key_order(
        local_storage, CollisionType::EDGE_EDGE, collisions);
    const size_t face_vert_count = append_type_in_key_order(
        local_storage, CollisionType::FACE_VERTEX, collisions);

    logger().trace(
        "edge-vert pairs {}, vert-vert pairs {}", edge_vert_count,
        vert_vert_count);
    logger().trace(
        "face-vert pairs {}, edge-edge pairs {}", face_vert_count,
        edge_edge_count);
}

void SmoothCollisionsBuilder<2>::merge(
    const tbb::enumerable_thread_specific<SmoothCollisionsBuilder<2>>&
        local_storage,
    SmoothCollisions& merged_collisions)
{
    // size up the hash items
    size_t total = 0;
    for (const auto& storage : local_storage) {
        total += storage.collisions.size();
    }

    merged_collisions.collisions.reserve(total);

    // Emit a canonical sequence that depends only on the contact set, not on
    // hash iteration or thread scheduling: vertex-vertex, then edge-vertex
    // collisions, each sorted by primitive ids. The collision objects
    // themselves (orientation, weight, dhat) are kept as built.
    auto& collisions = merged_collisions.collisions;
    const size_t vert_vert_count =
        append_in_key_order<SmoothCollisionTemplate<Point2, Point2>>(
            local_storage, &SmoothCollisionsBuilder<2>::vert_vert_2_to_id,
            collisions);
    const size_t edge_vert_count =
        append_in_key_order<SmoothCollisionTemplate<Edge2, Point2>>(
            local_storage, &SmoothCollisionsBuilder<2>::vert_edge_2_to_id,
            collisions);

    logger().trace(
        "edge-vert pairs {}, vert-vert pairs {}", edge_vert_count,
        vert_vert_count);
}

} // namespace ipc