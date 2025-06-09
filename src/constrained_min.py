import numpy as np


class BarrierObjective:
    """Helper class to create barrier objective functions."""

    def __init__(self, original_func, ineq_constraints, t):
        self.original_func = original_func
        self.ineq_constraints = ineq_constraints
        self.t = t

    def __call__(self, x_var, hessian=False):
        """Evaluate barrier objective function."""
        if hasattr(self.original_func, 'objective'):
            f_val = self.original_func.objective(x_var)
            f_grad = self.original_func.gradient(x_var)
            f_hess = self.original_func.hessian(x_var) if hessian else None
        else:
            if hessian:
                result = self.original_func(x_var, hessian=True)
                if len(result) == 3:
                    f_val, f_grad, f_hess = result
                else:
                    f_val, f_grad = result
                    f_hess = np.zeros((len(x_var), len(x_var)))
            else:
                f_val, f_grad = self.original_func(x_var, hessian=False)
                f_hess = None

        penalty_val = 0.0
        penalty_grad = np.zeros_like(x_var)
        penalty_hess = np.zeros((len(x_var), len(x_var))) if hessian else None

        for constraint in self.ineq_constraints:
            if hasattr(constraint, 'objective'):
                g_val = constraint.objective(x_var)
                g_grad = constraint.gradient(x_var)
                g_hess = constraint.hessian(x_var) if hessian else None
            else:
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

            # Add barrier penalty terms
            penalty_val += -np.log(-g_val)
            penalty_grad += -g_grad / g_val

            if hessian and g_hess is not None:
                first_term = np.outer(g_grad, g_grad) / (g_val**2)
                second_term = -g_hess / g_val
                penalty_hess += first_term + second_term

        # Combine: f(x) + (1/t) * penalty
        total_val = f_val + (1.0 / self.t) * penalty_val
        total_grad = f_grad + (1.0 / self.t) * penalty_grad

        if hessian:
            if f_hess is None or np.all(f_hess == 0):
                f_hess = np.zeros((len(x_var), len(x_var)))

            total_hess = f_hess + (1.0 / self.t) * penalty_hess
            regularization = max(1e-8, 1.0 / self.t) * np.eye(len(x_var))
            total_hess += regularization

            return total_val, total_grad, total_hess
        else:
            return total_val, total_grad

    def objective(self, x_var):
        """Function interface: get objective value only."""
        result = self.__call__(x_var, hessian=False)
        return result[0]

    def gradient(self, x_var):
        """Function interface: get gradient only."""
        result = self.__call__(x_var, hessian=False)
        return result[1]

    def hessian(self, x_var):
        """Function interface: get hessian only."""
        result = self.__call__(x_var, hessian=True)
        return result[2]


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs,
                x0, maximize=False):
    """Interior point method using log-barrier approach."""
    if maximize:
        def objective_to_min(x_var, hessian=False):
            if hessian:
                f, g, h = func(x_var, hessian=True)
                return -f, -g, -h
            else:
                f, g = func(x_var, hessian=False)
                return -f, -g
    else:
        objective_to_min = func

    x = np.array(x0, dtype=float)
    t = 1.0
    mu = 10.0

    path = [x.copy()]
    obj_values = []

    max_outer_iter = 1000
    tolerance = 1e-10

    for i, constraint in enumerate(ineq_constraints):
        if hasattr(constraint, 'objective'):
            g_val = constraint.objective(x)
            g_grad = constraint.gradient(x)
        else:
            g_val, g_grad = constraint(x)

        if g_val >= -1e-10:
            print(f"Warning: Initial point may not be strictly feasible for "
                  f"constraint {i}: g(x) = {g_val}")
            if g_val >= 0:
                x -= 0.01 * g_grad / (np.linalg.norm(g_grad) + 1e-12)

    for outer_iter in range(max_outer_iter):
        barrier_objective = BarrierObjective(
            objective_to_min, ineq_constraints, t)

        # Solve the inner Newton step
        result = newton_equality_constrained(
            barrier_objective, eq_constraints_mat, eq_constraints_rhs, x,
            ineq_constraints)

        if result is False:
            print(
                f"Inner Newton method failed at outer iteration {outer_iter}")
            break

        x = result

        real_obj_val = func.objective(x)
        obj_values.append(real_obj_val)
        path.append(x.copy())

        # Check stopping criterion
        duality_gap = len(ineq_constraints) / t
        if duality_gap < tolerance:
            break

        t *= mu

    return x, obj_values, path


def newton_equality_constrained(func, A, b, x0, ineq_constraints=None,
                                max_iter=1000, tol=1e-8):
    """Newton method with equality constraints using KKT system."""
    x = np.array(x0, dtype=float)

    for iteration in range(max_iter):
        f_grad = func.gradient(x)
        f_hess = func.hessian(x)

        if A is not None and A.size > 0:
            # Handle both 1D and 2D constraint matrices
            A_mat = np.atleast_2d(A) if A.ndim == 1 else A
            b_vec = np.atleast_1d(b)

            n = len(x)
            m = len(b_vec)

            kkt_lhs = np.block([
                [f_hess, A_mat.T],
                [A_mat, np.zeros((m, m))]
            ])

            kkt_rhs = np.concatenate([
                -f_grad,
                np.zeros(m)
            ])

            # Solve KKT system
            try:
                solution = np.linalg.solve(kkt_lhs, kkt_rhs)
                dx = solution[:n]
            except np.linalg.LinAlgError:
                return False
        else:
            try:
                dx = -np.linalg.solve(f_hess, f_grad)
            except np.linalg.LinAlgError:
                dx = -np.linalg.pinv(f_hess) @ f_grad

        lambda_current = np.sqrt(dx.T @ f_hess @ dx)
        if 0.5 * (lambda_current**2) < tol:
            return x  # Converged

        alpha = backtracking_line_search_with_feasibility(
            func, x, dx, f_grad, ineq_constraints)

        x = x + alpha * dx

    return x


def backtracking_line_search_with_feasibility(func, x, dx, grad,
                                              ineq_constraints=None,
                                              alpha_init=1.0, rho=0.5, c=1e-2):
    """Enhanced backtracking line search with feasibility check."""
    alpha = alpha_init
    f0 = func.objective(x)
    grad_dot_dx = np.dot(grad, dx)

    if grad_dot_dx >= 0:
        dx = -grad
        grad_dot_dx = -np.dot(grad, grad)

    for _ in range(50):
        x_new = x + alpha * dx

        feasible = True
        if ineq_constraints:
            for constraint in ineq_constraints:
                g_val = constraint.objective(x_new)
                if g_val >= 0:
                    feasible = False
                    break

        if not feasible:
            alpha *= rho
            continue

        try:
            f_new = func.objective(x_new)
            if not (np.isnan(f_new) or np.isinf(f_new)):
                if f_new <= f0 + c * alpha * grad_dot_dx:
                    return alpha
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            pass

        alpha *= rho
        if alpha < 1e-10:
            break

    return max(alpha, 1e-10)


def backtracking_line_search(func, x, dx, grad, alpha_init=1.0, rho=0.5,
                             c=1e-4):
    """Backtracking line search with Armijo condition."""
    alpha = alpha_init
    f0 = func.objective(x)
    grad_dot_dx = np.dot(grad, dx)

    if grad_dot_dx >= 0:
        dx = -grad
        grad_dot_dx = -np.dot(grad, grad)

    for _ in range(50):
        try:
            f_new = func.objective(x + alpha * dx)
            if not (np.isnan(f_new) or np.isinf(f_new)):
                if f_new <= f0 + c * alpha * grad_dot_dx:
                    return alpha
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            pass
        alpha *= rho
        if alpha < 1e-10:
            break

    return max(alpha, 1e-10)
