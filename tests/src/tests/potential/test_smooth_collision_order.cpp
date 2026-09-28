#include <tests/config.hpp>
#include <tests/utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <ipc/ipc.hpp>
#include <ipc/smooth_contact/smooth_contact_potential.hpp>

#include <igl/readCSV.h>
#include <tbb/global_control.h>
#include <tbb/info.h>

#include <algorithm>
#include <random>
#include <tuple>

using namespace ipc;

namespace {

// Everything that distinguishes one collision record from another.
using Record = std::tuple<
    CollisionType, index_t, index_t, double, double, bool, std::vector<index_t>>;

std::vector<Record> records(const SmoothCollisions& collisions)
{
    std::vector<Record> r;
    for (const auto& cc : collisions.collisions) {
        r.emplace_back(
            cc->type(), (*cc)[0], (*cc)[1], cc->weight, cc->dhat(),
            cc->is_active(), cc->vertex_ids());
    }
    return r;
}

struct Evaluation {
    double energy;
    Eigen::VectorXd gradient;
    Eigen::MatrixXd hessian;
};

// Serial evaluation: with one thread the summation follows the collision
// order, so equal orders must give bitwise-equal results.
Evaluation evaluate(
    const SmoothContactPotential& potential,
    const SmoothCollisions& collisions,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices)
{
    tbb::global_control serial(
        tbb::global_control::max_allowed_parallelism, 1);
    return { potential(collisions, mesh, vertices),
             potential.gradient(collisions, mesh, vertices),
             Eigen::MatrixXd(potential.hessian(collisions, mesh, vertices)) };
}

SmoothCollisions build(
    const Candidates& candidates,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const SmoothContactParameters& params,
    const SmoothCollisions& adaptive,
    const int threads)
{
    tbb::global_control control(
        tbb::global_control::max_allowed_parallelism, threads);
    SmoothCollisions collisions = adaptive; // carries the adaptive dhat
    collisions.build(candidates, mesh, vertices, params, true);
    return collisions;
}

} // namespace

TEST_CASE(
    "Smooth collisions have a canonical order",
    "[smooth_potential][determinism]")
{
    const bool is_2d = GENERATE(true, false);

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi edges, faces;
    double dhat;
    if (is_2d) {
        const std::string mesh_name =
            (tests::DATA_DIR / "gcp" / "nonlinear_solve_iter020.obj").string();
        REQUIRE(igl::readCSV(mesh_name + "-v.csv", vertices));
        REQUIRE(igl::readCSV(mesh_name + "-e.csv", edges));
        dhat = 3e-2;
    } else {
        REQUIRE(tests::load_mesh("two-cubes-close.ply", vertices, edges, faces));
        dhat = 1e-1;
    }
    CAPTURE(is_2d);

    CollisionMesh mesh;
    if (is_2d) {
        mesh = CollisionMesh(
            std::vector<bool>(vertices.rows(), true),
            std::vector<bool>(vertices.rows(), true), vertices, edges, faces);
    } else {
        mesh = CollisionMesh(
            CollisionMesh::construct_is_on_surface(vertices.rows(), edges),
            std::vector<bool>(vertices.rows(), true), vertices, edges, faces);
        vertices = mesh.vertices(vertices);
    }

    SmoothContactParameters params(dhat, 0.9, -0.05, 0.95, 0.05, 1);
    params.set_adaptive_dhat_ratio(1.5);
    SmoothCollisions adaptive;
    adaptive.compute_adaptive_dhat(
        mesh, vertices, params, make_default_broad_phase().get());

    Candidates candidates;
    candidates.build(mesh, vertices, dhat / 2);

    const SmoothCollisions reference =
        build(candidates, mesh, vertices, params, adaptive, 1);
    REQUIRE(reference.size() > 10);

    // The sequence is grouped by type and sorted by primitive ids within each
    // group, so it depends on the contact set only.
    const std::vector<CollisionType> type_order = is_2d
        ? std::vector<CollisionType> { CollisionType::VERTEX_VERTEX,
                                       CollisionType::EDGE_VERTEX }
        : std::vector<CollisionType> { CollisionType::VERTEX_VERTEX,
                                       CollisionType::EDGE_VERTEX,
                                       CollisionType::EDGE_EDGE,
                                       CollisionType::FACE_VERTEX };
    const auto rank = [&](const SmoothCollision& cc) {
        const auto it =
            std::find(type_order.begin(), type_order.end(), cc.type());
        REQUIRE(it != type_order.end());
        return std::make_tuple(it - type_order.begin(), cc[0], cc[1]);
    };
    for (size_t i = 1; i < reference.size(); i++) {
        CAPTURE(i);
        CHECK(rank(reference[i - 1]) <= rank(reference[i]));
    }

    const SmoothContactPotential potential(params);
    const Evaluation expected =
        evaluate(potential, reference, mesh, vertices);
    REQUIRE(expected.gradient.squaredNorm() > 0);

    // Permuting the insertion order (candidate order) or the thread count
    // changes neither the sequence nor, bitwise, the energy, gradient, or
    // Hessian evaluated from it.
    std::mt19937 rng(20260928);
    const int max_threads = std::max<int>(
        2, tbb::info::default_concurrency());
    for (int trial = 0; trial < 4; trial++) {
        CAPTURE(trial);
        Candidates permuted = candidates;
        std::shuffle(
            permuted.ev_candidates.begin(), permuted.ev_candidates.end(), rng);
        std::shuffle(
            permuted.ee_candidates.begin(), permuted.ee_candidates.end(), rng);
        std::shuffle(
            permuted.fv_candidates.begin(), permuted.fv_candidates.end(), rng);

        const SmoothCollisions collisions = build(
            permuted, mesh, vertices, params, adaptive,
            trial % 2 ? max_threads : 1);
        CHECK(records(collisions) == records(reference));

        const Evaluation actual =
            evaluate(potential, collisions, mesh, vertices);
        CHECK(actual.energy == expected.energy);
        CHECK(actual.gradient == expected.gradient);
        CHECK(actual.hessian == expected.hessian);
    }
}
