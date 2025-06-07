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

                # Check constraint satisfaction with small tolerance
                if g_val >= -1e-12:
                    penalty = 1e6 * (g_val + 1e-12)**2
                    if hessian:
                        penalty_hess = f_hess + 1e6 * np.eye(len(x_var))
                        return (f_val + penalty, f_grad, penalty_hess)
                    else:
                        return (f_val + penalty, f_grad)

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
        try:
            x_new = newton_equality_constrained(
                barrier_objective, eq_constraints_mat,
                eq_constraints_rhs, x)

            # Check if new point is valid
            if not np.any(np.isnan(x_new)) and not np.any(np.isinf(x_new)):
                x = x_new
            else:
                print(
                    f"Warning: Invalid solution at iteration {outer_iter}, "
                    f"using previous point")
                break

        except Exception as e:
            print(
                f"Warning: Newton step failed at iteration {outer_iter}: {e}")
            break

        # Store results
        try:
            obj_val, _ = func(x)
            if np.isnan(obj_val) or np.isinf(obj_val):
                print(
                    f"Warning: Invalid objective value at iteration "
                    f"{outer_iter}")
                break
            obj_values.append(obj_val)
            path.append(x.copy())
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
            print(
                f"Warning: Could not evaluate objective at iteration "
                f"{outer_iter}")
            break

        # Check stopping criterion
        duality_gap = len(ineq_constraints) / t
        if duality_gap < tolerance:
            break

        # Update barrier parameter
        t *= mu

    # Final step: ensure feasibility with proper projection
    # Check if any inequality constraints are violated
    violations = []
    for i, constraint in enumerate(ineq_constraints):
        g_val, _ = constraint(x)
        if g_val > 1e-12:
            violations.append(i)

    if violations:
        print(
            f"Warning: Projecting {len(violations)} constraint(s) back to "
            f"feasible region")

        # For bound constraints -x[i] <= 0 (i.e., x[i] >= 0), project to
        # boundary
        for i in violations:
            if i < len(x):  # Assume first len(x) constraints are bounds
                x[i] = 0.0

        # Re-satisfy equality constraints if they exist
        if eq_constraints_mat is not None and eq_constraints_rhs is not None:
            # Simple adjustment: distribute the violation among free variables
            eq_violation = eq_constraints_mat @ x - eq_constraints_rhs
            eq_violation_val = eq_violation if np.isscalar(
                eq_violation) else eq_violation[0]
            if abs(eq_violation_val) > 1e-12:
                # Find variables that are not at bounds
                free_vars = [i for i in range(len(x)) if i not in violations]
                if free_vars:
                    # Distribute violation equally among free variables
                    adjustment = eq_violation_val / len(free_vars)
                    for i in free_vars:
                        x[i] -= adjustment

    return x, obj_values, path


def newton_equality_constrained(func, A, b, x0, max_iter=50, tol=1e-8):
    """Newton method with equality constraints using KKT system."""
    x = np.array(x0, dtype=float)

    for i in range(max_iter):
        try:
            f_val, f_grad, f_hess = func(x, hessian=True)
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
            print("Failed to evaluate function")
            break

        # Check for NaN/Inf
        if np.any(np.isnan(f_grad)) or np.any(np.isinf(f_grad)):
            print("NaN/Inf detected in gradient")
            break

        # Ensure f_hess is a proper 2D array
        if f_hess is None or f_hess.ndim == 0:
            f_hess = np.eye(len(x)) * 1e-6
        elif f_hess.ndim == 1:
            f_hess = np.diag(f_hess)

        if A is not None:
            # Handle both 1D and 2D constraint matrices
            A_mat = np.atleast_2d(A)
            if A.ndim == 1:
                A_mat = A.reshape(1, -1)
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
                try:
                    solution = np.linalg.pinv(kkt_lhs) @ kkt_rhs
                    dx = solution[:n]
                except (np.linalg.LinAlgError, ValueError):
                    dx = -f_grad * 0.01
        else:
            # Unconstrained Newton step
            try:
                # Ensure matrix is well-conditioned
                cond_num = np.linalg.cond(f_hess)
                if cond_num > 1e12:
                    # Use regularized version
                    f_hess_reg = f_hess + (cond_num * 1e-16) * np.eye(len(x))
                    dx = -np.linalg.solve(f_hess_reg, f_grad)
                else:
                    dx = -np.linalg.solve(f_hess, f_grad)
            except np.linalg.LinAlgError:
                try:
                    dx = -np.linalg.pinv(f_hess) @ f_grad
                except (ValueError, TypeError, RuntimeError):
                    # Last resort: steepest descent with small step
                    dx = -f_grad / (np.linalg.norm(f_grad) + 1e-12) * 0.01

        # Check for valid step
        if np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
            dx = -f_grad * 0.001  # Very small steepest descent step

        # Line search
        alpha = backtracking_line_search(func, x, dx, f_grad)
        x_new = x + alpha * dx

        # Check if new point is valid
        if np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            break

        x = x_new

        # Check convergence
        if np.linalg.norm(f_grad) < tol:
            break

    return x


def backtracking_line_search(func, x, dx, grad, alpha_init=1.0, rho=0.5,
                             c=1e-4):
    """Backtracking line search with Armijo condition."""
    alpha = alpha_init

    try:
        f0, _ = func(x)
    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
        return 1e-6

    grad_dot_dx = np.dot(grad, dx)

    # If the search direction is not a descent direction, use steepest descent
    if grad_dot_dx >= 0:
        dx = -grad
        grad_dot_dx = -np.dot(grad, grad)

    for _ in range(50):  # Limit iterations
        try:
            f_new, _ = func(x + alpha * dx)
            if not (np.isnan(f_new) or np.isinf(f_new)):
                if f_new <= f0 + c * alpha * grad_dot_dx:
                    return alpha
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
            pass
        alpha *= rho
        if alpha < 1e-10:
            break

    return max(alpha, 1e-10)
