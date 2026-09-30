#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <ipc/tangent/closest_point.hpp>

#include <finitediff.hpp>

using namespace ipc;

TEST_CASE(
    "Point-triangle closest point", "[friction][point-triangle][closest_point]")
{
    Eigen::Vector3d t0(-1, 0, 1), t1(1, 0, 1), t2(0, 0, -1);
    Eigen::Vector2d expected_coords(0.5, 0.5);
    Eigen::Vector3d p =
        t0 + expected_coords[0] * (t1 - t0) + expected_coords[1] * (t2 - t0);
    // p = 1 * t0 + u * t1 - u * t0 + v * t2 - v * t0
    //   = (1 - u - v) * t0 + u * t1 + v * t2
    //   =  w * t0 + u * t1 + v * t2

    Eigen::Vector2d barycentric_coords =
        point_triangle_closest_point(p, t0, t1, t2);
    Eigen::Vector3d p_actual = t0 + barycentric_coords[0] * (t1 - t0)
        + barycentric_coords[1] * (t2 - t0);
    CAPTURE(barycentric_coords);
    CHECK((p - p_actual).norm() == Catch::Approx(0).margin(1e-12));

    // test Jacobian
    Eigen::Matrix<double, 2, 12> J =
        point_triangle_closest_point_jacobian(p, t0, t1, t2);

    Vector12d x;
    x << p, t0, t1, t2;

    Eigen::MatrixXd J_FD;
    fd::finite_jacobian(
        x,
        [](const Eigen::VectorXd& _x) {
            return point_triangle_closest_point(
                _x.segment<3>(0), _x.segment<3>(3), _x.segment<3>(6),
                _x.segment<3>(9));
        },
        J_FD);

    CHECK(fd::compare_jacobian(J, J_FD));

    // test Hessian: one 12x12 block per barycentric coordinate, each the
    // Jacobian of that coordinate's row of J.
    const std::array<Eigen::Matrix<double, 12, 12>, 2> H =
        point_triangle_closest_point_hessian(p, t0, t1, t2);

    for (int c = 0; c < 2; c++) {
        CAPTURE(c);
        Eigen::MatrixXd H_FD;
        fd::finite_jacobian(
            x,
            [c](const Eigen::VectorXd& _x) -> Eigen::VectorXd {
                return point_triangle_closest_point_jacobian(
                           _x.segment<3>(0), _x.segment<3>(3), _x.segment<3>(6),
                           _x.segment<3>(9))
                    .row(c)
                    .transpose();
            },
            H_FD);

        CHECK(fd::compare_jacobian(H[c], H_FD));
    }
}

TEST_CASE("Edge-edge closest point", "[friction][edge-edge][closest_point]")
{
    Eigen::Vector3d ea0(-1, 0, 0), ea1(1, 0, 0), eb0(0, 0, -1), eb1(0, 0, 1);

    Eigen::Vector2d barycentric_coords =
        edge_edge_closest_point(ea0, ea1, eb0, eb1);
    CAPTURE(barycentric_coords);
    // Perpendicular edges centered on the same axis meet at their midpoints,
    // so the answer is exactly (0.5, 0.5) with no rounding to hide behind.
    CHECK(barycentric_coords[0] == Catch::Approx(0.5).epsilon(0).margin(1e-15));
    CHECK(barycentric_coords[1] == Catch::Approx(0.5).epsilon(0).margin(1e-15));

    // test Jacobian
    Eigen::Matrix<double, 2, 12> J =
        edge_edge_closest_point_jacobian(ea0, ea1, eb0, eb1);

    Vector12d x;
    x << ea0, ea1, eb0, eb1;

    Eigen::MatrixXd J_FD;
    fd::finite_jacobian(
        x,
        [](const Eigen::VectorXd& _x) {
            return edge_edge_closest_point(
                _x.segment<3>(0), _x.segment<3>(3), _x.segment<3>(6),
                _x.segment<3>(9));
        },
        J_FD);

    CHECK(fd::compare_jacobian(J, J_FD));

    // test Hessian: one 12x12 block per barycentric coordinate, each the
    // Jacobian of that coordinate's row of J.
    const std::array<Eigen::Matrix<double, 12, 12>, 2> H =
        edge_edge_closest_point_hessian(ea0, ea1, eb0, eb1);

    for (int c = 0; c < 2; c++) {
        CAPTURE(c);
        Eigen::MatrixXd H_FD;
        fd::finite_jacobian(
            x,
            [c](const Eigen::VectorXd& _x) -> Eigen::VectorXd {
                return edge_edge_closest_point_jacobian(
                           _x.segment<3>(0), _x.segment<3>(3), _x.segment<3>(6),
                           _x.segment<3>(9))
                    .row(c)
                    .transpose();
            },
            H_FD);

        CHECK(fd::compare_jacobian(H[c], H_FD));
    }
}

TEST_CASE("Point-edge closest point", "[friction][point-edge][closest_point]")
{
    Eigen::Vector3d p(0, 1, 0), e0(-1, 0, 0), e1(1, 0, 0);

    double alpha = point_edge_closest_point(p, e0, e1);
    CHECK(alpha == Catch::Approx(0.5));

    // test Jacobian
    VectorMax9d J = point_edge_closest_point_jacobian(p, e0, e1);

    Vector9d x;
    x << p, e0, e1;

    Eigen::VectorXd J_FD;
    fd::finite_gradient(
        x,
        [](const Eigen::VectorXd& _x) {
            return point_edge_closest_point(
                _x.segment<3>(0), _x.segment<3>(3), _x.segment<3>(6));
        },
        J_FD);

    CHECK(fd::compare_gradient(J, J_FD));

    // test Hessian: the closest point is a scalar here, so its Hessian is the
    // Jacobian of the gradient checked above.
    const Eigen::Matrix<double, 9, 9> H =
        point_edge_closest_point_hessian(p, e0, e1);

    Eigen::MatrixXd H_FD;
    fd::finite_jacobian(
        x,
        [](const Eigen::VectorXd& _x) -> Eigen::VectorXd {
            return point_edge_closest_point_jacobian(
                _x.segment<3>(0), _x.segment<3>(3), _x.segment<3>(6));
        },
        H_FD);

    CHECK(fd::compare_jacobian(H, H_FD));
}

TEST_CASE(
    "Point-edge closest point in 2D",
    "[friction][point-edge][closest_point][2D]")
{
    Eigen::Vector2d p(0, 1), e0(-1, 0), e1(1, 0);

    double alpha = point_edge_closest_point(p, e0, e1);
    CHECK(alpha == Catch::Approx(0.5));

    // test Jacobian
    VectorMax9d J = point_edge_closest_point_jacobian(p, e0, e1);

    Vector6d x;
    x << p, e0, e1;

    Eigen::VectorXd J_FD;
    fd::finite_gradient(
        x,
        [](const Eigen::VectorXd& _x) {
            return point_edge_closest_point(
                _x.segment<2>(0), _x.segment<2>(2), _x.segment<2>(4));
        },
        J_FD);

    CHECK(fd::compare_gradient(J, J_FD));

    // test Hessian: covers the 2D branch of the kernel, which the 3D case
    // above never reaches.
    const Eigen::Matrix<double, 6, 6> H =
        point_edge_closest_point_hessian(p, e0, e1);

    Eigen::MatrixXd H_FD;
    fd::finite_jacobian(
        x,
        [](const Eigen::VectorXd& _x) -> Eigen::VectorXd {
            return point_edge_closest_point_jacobian(
                _x.segment<2>(0), _x.segment<2>(2), _x.segment<2>(4));
        },
        H_FD);

    CHECK(fd::compare_jacobian(H, H_FD));
}

TEST_CASE(
    "Point-edge closest point agrees whether or not the dimension is known at "
    "compile time",
    "[friction][point-edge][closest_point]")
{
    // The front ends dispatch on `dim_v<Derived>` when the argument type
    // carries its size and fall back to a runtime branch on `size()` when it
    // does not. Every other test here passes a fixed-size vector, so only the
    // `if constexpr` arms ever run. A VectorMax3d holds the same numbers but
    // keeps its size at runtime, taking the fallback instead; both arms reach
    // the same kernel, so any difference is a dispatch bug.
    const int dim = GENERATE(2, 3);
    CAPTURE(dim);

    const Eigen::VectorXd p = Eigen::VectorXd::LinSpaced(dim, 0.25, 1.0);
    const Eigen::VectorXd e0 = Eigen::VectorXd::LinSpaced(dim, -1.0, 0.5);
    const Eigen::VectorXd e1 = Eigen::VectorXd::LinSpaced(dim, 0.75, -0.5);

    const VectorMax3d p_dyn = p, e0_dyn = e0, e1_dyn = e1;

    if (dim == 2) {
        const Eigen::Vector2d a = p, b = e0, c = e1;
        CHECK(
            point_edge_closest_point(p_dyn, e0_dyn, e1_dyn)
            == point_edge_closest_point(a, b, c));
        CHECK(
            point_edge_closest_point_jacobian(p_dyn, e0_dyn, e1_dyn)
            == point_edge_closest_point_jacobian(a, b, c));
        CHECK(
            point_edge_closest_point_hessian(p_dyn, e0_dyn, e1_dyn)
            == point_edge_closest_point_hessian(a, b, c));
    } else {
        const Eigen::Vector3d a = p, b = e0, c = e1;
        CHECK(
            point_edge_closest_point(p_dyn, e0_dyn, e1_dyn)
            == point_edge_closest_point(a, b, c));
        CHECK(
            point_edge_closest_point_jacobian(p_dyn, e0_dyn, e1_dyn)
            == point_edge_closest_point_jacobian(a, b, c));
        CHECK(
            point_edge_closest_point_hessian(p_dyn, e0_dyn, e1_dyn)
            == point_edge_closest_point_hessian(a, b, c));
    }
}

namespace {
// Cramer's rule with Kahan's determinant and no refinement: the kernel
// `solve_spd_2x2` used before it learned to refine.
Eigen::Vector2d cramer_2x2(const Eigen::Matrix2d& A, const Eigen::Vector2d& b)
{
    const double bc = A(0, 1) * A(1, 0);
    const double bc_err = ipc::numext::fma(A(0, 1), A(1, 0), -bc);
    const double det = ipc::numext::fma(A(0, 0), A(1, 1), -bc) - bc_err;
    const double inv_det = det > 0 ? 1.0 / det : 0.0;
    return Eigen::Vector2d(
        (A(1, 1) * b[0] - A(0, 1) * b[1]) * inv_det,
        (A(0, 0) * b[1] - A(1, 0) * b[0]) * inv_det);
}

double relative_residual(
    const Eigen::Matrix2d& A,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& x)
{
    return (A * x - b).norm() / (A.norm() * x.norm() + b.norm());
}
} // namespace

TEST_CASE(
    "Nearly parallel 2x2 solve stays accurate",
    "[friction][edge-edge][closest_point][solve_spd_2x2]")
{
    // The Gram system of the nearly parallel edge pair from a debug-build
    // failure of the semi-implicit stiffness (edges ~6e-4 rad from parallel,
    // cond(A) ~ 4e7). Unrefined Cramer's rule leaves an absolute residual of
    // 4.3e-9 without hardware FMA, a relative one of 2.1e-10 (1.4e-10 with
    // FMA), above the 1e-10 bound the debug assertion enforces.
    Eigen::Matrix2d A;
    A << 0.067896765590525515, 1.1037422437870266, 1.1037422437870266,
        17.942643306292943;
    const Eigen::Vector2d b(0.62082203665333358, 10.092200801559782);

    const Eigen::Vector2d x_cramer = cramer_2x2(A, b);
    const Eigen::Vector2d x = ipc::detail::solve_spd_2x2<double>(A, b);

    // Reference: the pivoted LDLT solve this code used originally, and the
    // exact solution in extended precision.
    const Eigen::Vector2d x_ldlt = A.ldlt().solve(b);
    const Eigen::Vector2d x_exact =
        (Eigen::Matrix2<long double>(A.cast<long double>()).inverse()
         * b.cast<long double>())
            .cast<double>();

    CAPTURE(x_cramer.transpose(), x.transpose(), x_ldlt.transpose());

    // The failure this guards against is really there without refinement ...
    CHECK(
        relative_residual(A, b, x_cramer)
        > ipc::detail::CLOSEST_POINT_RESIDUAL_TOL<double>);
    // ... and refinement brings the residual below the debug bound, to the
    // rounding level of the LDLT solve. (The forward error stays at
    // cond(A) * eps, as for any backward-stable solve, so it is only bounded.)
    CHECK(
        relative_residual(A, b, x)
        < ipc::detail::CLOSEST_POINT_RESIDUAL_TOL<double>);
    CHECK(relative_residual(A, b, x) < 1e-15);
    CHECK((x - x_exact).norm() < 1e-7);
}

TEST_CASE(
    "2x2 solve keeps the Cramer bits where Cramer is accurate",
    "[friction][edge-edge][closest_point][solve_spd_2x2]")
{
    // Well-conditioned Gram systems: the residual is at rounding level, so the
    // refinement must not fire and the result must equal Cramer's rule bit for
    // bit (the semi-implicit smoke scenes rely on this).
    const std::vector<std::pair<Eigen::Matrix2d, Eigen::Vector2d>> systems = {
        { (Eigen::Matrix2d() << 2.0, 0.5, 0.5, 1.0).finished(),
          Eigen::Vector2d(1.0, -3.0) },
        { (Eigen::Matrix2d() << 1.0, 0.0, 0.0, 1.0).finished(),
          Eigen::Vector2d(0.3, 0.7) },
        { (Eigen::Matrix2d() << 4.25, -1.75, -1.75, 0.8125).finished(),
          Eigen::Vector2d(2.5, 0.125) },
        { (Eigen::Matrix2d() << 1.3, 0.2, 0.2, 5.1).finished(),
          Eigen::Vector2d(-0.4, 0.9) },
    };

    for (const auto& [A, b] : systems) {
        CAPTURE(A, b);
        const Eigen::Vector2d x = ipc::detail::solve_spd_2x2<double>(A, b);
        const Eigen::Vector2d x_cramer = cramer_2x2(A, b);
        CHECK(x[0] == x_cramer[0]);
        CHECK(x[1] == x_cramer[1]);
        CHECK(relative_residual(A, b, x) < 1e-15);
    }

    // Randomized, mildly conditioned Gram matrices A = B Bᵀ.
    srand(1234);
    for (int i = 0; i < 1000; i++) {
        const Eigen::Matrix2d B = Eigen::Matrix2d::Random();
        const Eigen::Matrix2d A =
            B * B.transpose() + 0.5 * Eigen::Matrix2d::Identity();
        const Eigen::Vector2d b = Eigen::Vector2d::Random();
        const Eigen::Vector2d x = ipc::detail::solve_spd_2x2<double>(A, b);
        const Eigen::Vector2d x_cramer = cramer_2x2(A, b);
        CHECK(x[0] == x_cramer[0]);
        CHECK(x[1] == x_cramer[1]);
    }
}
