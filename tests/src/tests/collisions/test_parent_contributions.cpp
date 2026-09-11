// Parent contributions (sdast9 fork): every candidate primitive pair that
// builds a collision is recorded on it with the weight it added, so callers
// can key per-collision quantities (stiffness_scale) on the parent and have
// them survive closest-feature switches.
#include <catch2/catch_test_macros.hpp>

#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/broad_phase/hash_grid.hpp>

#include <cmath>

using namespace ipc;

namespace {

// A 2D corner: edge 0 = (v0, v1) along -x, edge 1 = (v1, v2) down -y, and a
// free vertex v3 at (x, y). For x < 0 the free vertex is in edge 0's interior
// region; for x > 0 it is beyond the corner v1 for both edges.
CollisionMesh corner_mesh(const double x, const double y = .2)
{
    Eigen::MatrixXd vertices(4, 2);
    vertices << -1, 0, //
        0, 0,          //
        0, -1,         //
        x, y;
    Eigen::MatrixXi edges(2, 2);
    edges << 0, 1, //
        1, 2;
    CollisionMesh mesh(vertices, edges);
    mesh.can_collide = [](size_t a, size_t b) { return (a == 3) != (b == 3); };
    return mesh;
}

double parent_weight_sum(const NormalCollision& c)
{
    double sum = 0;
    for (const auto& p : c.parents) {
        sum += p.weight;
    }
    return sum;
}

} // namespace

TEST_CASE(
    "Parent contributions follow candidates through the builder",
    "[collisions][parents]")
{
    constexpr double dhat = 1.0;
    HashGrid broad_phase;

    SECTION(
        "beyond the corner: one vertex-vertex collision with two edge parents")
    {
        const CollisionMesh mesh = corner_mesh(.1);
        NormalCollisions collisions;
        collisions.build(mesh, mesh.rest_positions(), dhat, 0, &broad_phase);

        REQUIRE(collisions.size() == 1);
        REQUIRE(collisions.is_vertex_vertex(0));
        const NormalCollision& vv = collisions[0];
        CHECK(vv.weight == 2.0);
        REQUIRE(vv.parents.size() == 2);
        CHECK(parent_weight_sum(vv) == vv.weight);
        for (const auto& p : vv.parents) {
            CHECK(p.type == ParentContribution::Type::EdgeVertex);
            CHECK(p.id1 == 3);
            CHECK(p.weight == 1.0);
        }
        CHECK(vv.parents[0].id0 != vv.parents[1].id0);
    }

    SECTION(
        "in the interior region: an edge-vertex collision and a single-parent vertex-vertex")
    {
        const CollisionMesh mesh = corner_mesh(-.3);
        NormalCollisions collisions;
        collisions.build(mesh, mesh.rest_positions(), dhat, 0, &broad_phase);

        REQUIRE(collisions.size() == 2);
        for (size_t i = 0; i < collisions.size(); ++i) {
            const NormalCollision& c = collisions[i];
            REQUIRE(c.parents.size() == 1);
            CHECK(c.parents[0].type == ParentContribution::Type::EdgeVertex);
            CHECK(c.parents[0].id1 == 3);
            CHECK(parent_weight_sum(c) == c.weight);
            // The edge-vertex collision belongs to edge 0, the vertex-vertex
            // (with the corner v1) to edge 1.
            CHECK(c.parents[0].id0 == (collisions.is_edge_vertex(i) ? 0 : 1));
        }
    }

    SECTION("stiffness_scale defaults to one and is untouched by the parents")
    {
        const CollisionMesh mesh = corner_mesh(.1);
        NormalCollisions collisions;
        collisions.build(mesh, mesh.rest_positions(), dhat, 0, &broad_phase);
        for (size_t i = 0; i < collisions.size(); ++i) {
            CHECK(collisions[i].stiffness_scale == 1.0);
        }
    }

    SECTION("area weighting keeps the parents' weights summing to the total")
    {
        const CollisionMesh mesh = corner_mesh(.1);
        NormalCollisions collisions;
        collisions.set_use_area_weighting(true);
        collisions.build(mesh, mesh.rest_positions(), dhat, 0, &broad_phase);
        REQUIRE(collisions.size() >= 1);
        for (size_t i = 0; i < collisions.size(); ++i) {
            const NormalCollision& c = collisions[i];
            REQUIRE(!c.parents.empty());
            CHECK(
                std::abs(parent_weight_sum(c) - c.weight)
                <= 1e-15 * std::abs(c.weight));
        }
    }
}
