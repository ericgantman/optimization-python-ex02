"""
Unconstrained minimization module
Contains utilities needed for constrained optimization
"""
import numpy as np


def line_search_tool(func, x, p, grad, c1=1e-4, c2=0.9, max_iter=50):
    """
    Simple backtracking line search.

    Args:
        func: Objective function
        x: Current point
        p: Search direction
        grad: Current gradient
        c1: Armijo condition parameter
        c2: Curvature condition parameter (not used in backtracking)
        max_iter: Maximum iterations

    Returns:
        alpha: Step size
    """
    alpha = 1.0
    func_result = func(x)
    f0 = func_result[0] if isinstance(func_result, tuple) else func_result

    for _ in range(max_iter):
        try:
            func_result_new = func(x + alpha * p)
            f_new = func_result_new[0] if isinstance(
                func_result_new, tuple) else func_result_new
            # Armijo condition
            if f_new <= f0 + c1 * alpha * np.dot(grad, p):
                return alpha
        except (ValueError, OverflowError, ZeroDivisionError):
            pass
        alpha *= 0.5

    return alpha


def newton_method(func, x0, eq_constraints_mat=None, eq_constraints_rhs=None,
                  max_iter=100, tolerance=1e-8):
    """
    Newton's method for unconstrained or equality-constrained optimization.

    Args:
        func: Objective function with interface (x, hessian=False)
        x0: Initial point
        eq_constraints_mat: Equality constraint matrix A
        eq_constraints_rhs: Equality constraint RHS b
        max_iter: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        x: Solution
    """
    x = np.array(x0, dtype=float)

    for iteration in range(max_iter):
        f_val, f_grad, f_hess = func(x, hessian=True)

        if eq_constraints_mat is not None:
            # Solve equality-constrained problem using KKT system
            n = len(x)
            A = eq_constraints_mat
            b = eq_constraints_rhs

            if A.ndim == 1:
                A = A.reshape(1, -1)
                b = np.array([b]) if np.isscalar(b) else b

            m = A.shape[0]

            # Build KKT matrix: [H A^T]
            #                   [A  0 ]
            kkt_matrix = np.zeros((n + m, n + m))
            kkt_matrix[:n, :n] = f_hess
            kkt_matrix[:n, n:] = A.T
            kkt_matrix[n:, :n] = A

            # Build RHS: [-grad]
            #            [b-Ax ]
            rhs = np.zeros(n + m)
            rhs[:n] = -f_grad
            rhs[n:] = b - A @ x

            # Solve system
            try:
                sol = np.linalg.solve(kkt_matrix, rhs)
                dx = sol[:n]
            except np.linalg.LinAlgError:
                # Fallback for singular matrix
                sol = np.linalg.pinv(kkt_matrix) @ rhs
                dx = sol[:n]
        else:
            # Unconstrained case
            try:
                dx = -np.linalg.solve(f_hess, f_grad)
            except np.linalg.LinAlgError:
                dx = -np.linalg.pinv(f_hess) @ f_grad

        # Line search
        alpha = line_search_tool(func, x, dx, f_grad)
        x = x + alpha * dx

        # Check convergence
        if np.linalg.norm(f_grad) < tolerance:
            break

    return x
