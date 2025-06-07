import numpy as np


def interior_pt(func, ineq_constraints, eq_constraints_mat,
                eq_constraints_rhs, x0):
    """
    Interior point method using log-barrier approach.

    Args:
        func: Objective function with interface
            (x, hessian=False) -> (f_val, f_grad, f_hess)
        ineq_constraints: List of inequality constraint functions g_i(x) <= 0
        eq_constraints_mat: Matrix A for equality constraints Ax = b
            (can be None)
        eq_constraints_rhs: Vector b for equality constraints Ax = b
            (can be None)
        x0: Initial interior point

    Returns:
        tuple: (final_x, objective_values_list, path_points_list)
    """
    x = np.array(x0, dtype=float)
    t = 1.0  # Initial barrier parameter
    mu = 10.0  # Barrier parameter multiplier

    path = [x.copy()]
    obj_values = []

    max_outer_iter = 50
    tolerance = 1e-8

    # Check initial feasibility
    for i, constraint in enumerate(ineq_constraints):
        g_val, _ = constraint(x)
        if g_val >= -1e-10:
            print(
                f"Warning: Initial point may not be strictly feasible for "
                f"constraint {i}: g(x) = {g_val}")
            # Try to move the point slightly into the feasible region
            if g_val >= 0:
                # Move point away from constraint boundary
                g_grad = constraint(x)[1]
                x = x - 0.01 * g_grad / (np.linalg.norm(g_grad) + 1e-12)

    for outer_iter in range(max_outer_iter):
        # Create barrier function
        def barrier_objective(x_var, hessian=False):
            # Get original objective
            if hessian:
                result = func(x_var, hessian=True)
                if len(result) == 3:
                    f_val, f_grad, f_hess = result
                else:
                    f_val, f_grad = result
                    f_hess = np.zeros((len(x_var), len(x_var)))
            else:
                f_val, f_grad = func(x_var, hessian=False)
                f_hess = None

            # Add log-barrier terms
            barrier_val = 0.0
            barrier_grad = np.zeros_like(x_var)
            barrier_hess = np.zeros(
                (len(x_var), len(x_var))) if hessian else None

            for constraint in ineq_constraints:
                if hessian:
                    g_result = constraint(x_var, hessian=True)
                    if len(g_result) == 3:
                        g_val, g_grad, g_hess = g_result
                    else:
                        g_val, g_grad = g_result
                        g_hess = np.zeros((len(x_var), len(x_var)))
                else:
                    g_val, g_grad = constraint(x_var, hessian=False)
                    g_hess = None

                # Check constraint satisfaction
                if g_val >= 0:
                    if hessian:
                        return (np.inf, np.full_like(x_var, np.inf), None)
                    else:
                        return (np.inf, np.full_like(x_var, np.inf))

                # Add log-barrier terms: -log(-g(x))
                barrier_val -= np.log(-g_val)
                barrier_grad -= g_grad / g_val

                if hessian and g_hess is not None:
                    barrier_hess -= (np.outer(g_grad, g_grad) /
                                     (g_val**2) + g_hess / g_val)

            # Combine original objective with barrier
            total_val = t * f_val + barrier_val
            total_grad = t * f_grad + barrier_grad

            if hessian:
                # Handle case where original function has zero/None hessian
                # (like LP)
                if f_hess is None or np.all(f_hess == 0):
                    f_hess = np.zeros((len(x_var), len(x_var)))

                total_hess = t * f_hess + barrier_hess

                # Add regularization for numerical stability
                regularization = max(1e-8, 1.0 / t) * np.eye(len(x_var))
                total_hess += regularization

                return total_val, total_grad, total_hess
            else:
                return total_val, total_grad

        # Solve constrained Newton step
        x = newton_equality_constrained(
            barrier_objective, eq_constraints_mat, eq_constraints_rhs, x)

        # Store results
        obj_val, _ = func(x)
        obj_values.append(obj_val)
        path.append(x.copy())

        # Check stopping criterion
        duality_gap = len(ineq_constraints) / t
        if duality_gap < tolerance:
            break

        # Update barrier parameter
        t *= mu

    return x, obj_values, path


def newton_equality_constrained(func, A, b, x0, max_iter=50, tol=1e-8):
    """Newton method with equality constraints using KKT system."""
    x = np.array(x0, dtype=float)

    for _ in range(max_iter):
        f_val, f_grad, f_hess = func(x, hessian=True)

        # Check for convergence
        if np.linalg.norm(f_grad) < tol:
            break

        if A is not None:
            # Handle both 1D and 2D constraint matrices
            A_mat = np.atleast_2d(A) if A.ndim == 1 else A
            b_vec = np.atleast_1d(b)

            n = len(x)
            m = len(b_vec)

            # Build KKT system: [H  A^T] [dx] = [-grad]
            #                   [A   0 ] [dλ]   [b-Ax]
            kkt_lhs = np.zeros((n + m, n + m))
            kkt_lhs[:n, :n] = f_hess
            kkt_lhs[:n, n:] = A_mat.T
            kkt_lhs[n:, :n] = A_mat

            kkt_rhs = np.zeros(n + m)
            kkt_rhs[:n] = -f_grad
            kkt_rhs[n:] = b_vec - A_mat @ x

            # Solve KKT system
            try:
                solution = np.linalg.solve(kkt_lhs, kkt_rhs)
                dx = solution[:n]
            except np.linalg.LinAlgError:
                solution = np.linalg.pinv(kkt_lhs) @ kkt_rhs
                dx = solution[:n]
        else:
            # Unconstrained Newton step
            try:
                dx = -np.linalg.solve(f_hess, f_grad)
            except np.linalg.LinAlgError:
                dx = -np.linalg.pinv(f_hess) @ f_grad

        # Line search
        alpha = backtracking_line_search(func, x, dx, f_grad)
        x = x + alpha * dx

    return x


def backtracking_line_search(func, x, dx, grad, alpha_init=1.0, rho=0.5,
                             c=1e-4):
    """Backtracking line search with Armijo condition."""
    alpha = alpha_init
    f0, _ = func(x)
    grad_dot_dx = np.dot(grad, dx)

    # If not a descent direction, use steepest descent
    if grad_dot_dx >= 0:
        dx = -grad
        grad_dot_dx = -np.dot(grad, grad)

    for _ in range(50):
        try:
            f_new, _ = func(x + alpha * dx)
            if not (np.isnan(f_new) or np.isinf(f_new)):
                if f_new <= f0 + c * alpha * grad_dot_dx:
                    return alpha
        except:
            pass
        alpha *= rho
        if alpha < 1e-10:
            break

    return max(alpha, 1e-10)
