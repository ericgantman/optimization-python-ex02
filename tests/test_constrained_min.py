from examples import (qp_objective, qp_ineq_constraints,
                      qp_eq_constraints_mat, qp_eq_constraints_rhs,
                      lp_objective, lp_ineq_constraints)
from src.utils import (plot_3d_simplex_with_path,
                       plot_2d_feasible_region_with_path,
                       save_and_show_plot)
from src.constrained_min import interior_pt
import unittest
import numpy as np


class TestConstrainedMin(unittest.TestCase):

    def test_qp(self):
        """Test quadratic programming problem: min x² + y² + (z+1)²"""
        print("\n=== Testing QP Problem ===")

        x0 = np.array([0.1, 0.2, 0.7])
        x_final, obj_vals, path = interior_pt(
            qp_objective,
            qp_ineq_constraints,
            qp_eq_constraints_mat,
            qp_eq_constraints_rhs,
            x0
        )

        final_obj = qp_objective(x_final)[0]
        print(f"Final solution: x={x_final}")
        print(f"Final objective value: {final_obj:.6f}")
        print(f"Constraint x+y+z=1: {np.sum(x_final):.6f}")
        print(
            f"Non-negativity constraints: x={x_final[0]:.6f}, "
            f"y={x_final[1]:.6f}, z={x_final[2]:.6f}")

        # Generate plots
        self._plot_qp_results(path, obj_vals, x_final)

        self.assertAlmostEqual(np.sum(x_final), 1.0, places=4)
        self.assertTrue(np.all(x_final >= -1e-6))

    def test_lp(self):
        """Test linear programming problem: max x + y"""
        print("\n=== Testing LP Problem ===")

        x0 = np.array([0.5, 0.75])
        x_final, obj_vals, path = interior_pt(
            lp_objective,
            lp_ineq_constraints,
            None,
            None,
            x0
        )

        final_obj_min = lp_objective(x_final)[0]
        final_obj_max = -final_obj_min
        print(f"Final solution: x={x_final}")
        print(f"Final objective value (max x+y): {final_obj_max:.6f}")
        print("Constraint values at final solution:")
        for i, constraint in enumerate(lp_ineq_constraints):
            g_val, _ = constraint(x_final)
            print(f"  g_{i+1}(x) = {g_val:.6f} (should be ≤ 0)")

        # Generate plots
        self._plot_lp_results(path, obj_vals, x_final)

        self.assertGreater(final_obj_max, 0.8)
        for i, constraint in enumerate(lp_ineq_constraints):
            g_val, _ = constraint(x_final)
            self.assertLessEqual(
                g_val, 1e-6, f"Constraint {i+1} violated: g(x) = {g_val}")

    def _plot_qp_results(self, path, obj_vals, x_final):
        """Generate QP plots using utils."""
        fig = plot_3d_simplex_with_path(path, obj_vals, x_final,
                                        "QP: Feasible Region and Central Path")
        save_and_show_plot(fig, 'qp_results.png')

    def _plot_lp_results(self, path, obj_vals, x_final):
        """Generate LP plots using utils."""
        fig = plot_2d_feasible_region_with_path(
            path, obj_vals, x_final,
            "LP: Feasible Region and Central Path")
        save_and_show_plot(fig, 'lp_results.png')


if __name__ == '__main__':
    unittest.main(verbosity=2)
