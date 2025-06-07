import numpy as np

# =============================================================================
# QP Problem: min x² + y² + (z+1)² subject to x+y+z=1, x,y,z ≥ 0
# =============================================================================


def qp_objective(x, hessian=False):
    """Objective function: x² + y² + (z+1)²"""
    f_val = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    f_grad = np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])

    if hessian:
        f_hess = 2 * np.eye(3)
        return f_val, f_grad, f_hess
    return f_val, f_grad


def qp_ineq_constraint_x(x, hessian=False):
    """Inequality constraint: -x ≤ 0 (i.e., x ≥ 0)"""
    g_val = -x[0]
    g_grad = np.array([-1.0, 0.0, 0.0])

    if hessian:
        g_hess = np.zeros((3, 3))
        return g_val, g_grad, g_hess
    return g_val, g_grad


def qp_ineq_constraint_y(x, hessian=False):
    """Inequality constraint: -y ≤ 0 (i.e., y ≥ 0)"""
    g_val = -x[1]
    g_grad = np.array([0.0, -1.0, 0.0])

    if hessian:
        g_hess = np.zeros((3, 3))
        return g_val, g_grad, g_hess
    return g_val, g_grad


def qp_ineq_constraint_z(x, hessian=False):
    """Inequality constraint: -z ≤ 0 (i.e., z ≥ 0)"""
    g_val = -x[2]
    g_grad = np.array([0.0, 0.0, -1.0])

    if hessian:
        g_hess = np.zeros((3, 3))
        return g_val, g_grad, g_hess
    return g_val, g_grad


# QP constraint data
qp_ineq_constraints = [qp_ineq_constraint_x,
                       qp_ineq_constraint_y, qp_ineq_constraint_z]
qp_eq_constraints_mat = np.array(
    [1.0, 1.0, 1.0])  # [1, 1, 1] for x + y + z = 1
qp_eq_constraints_rhs = 1.0

# =============================================================================
# LP Problem: max x + y subject to constraints
# =============================================================================


def lp_objective(x, hessian=False):
    """Objective function: -(x + y) for minimization
    (equivalent to max x + y)"""
    f_val = -(x[0] + x[1])
    f_grad = np.array([-1.0, -1.0])

    if hessian:
        f_hess = np.zeros((2, 2))
        return f_val, f_grad, f_hess
    return f_val, f_grad


def lp_ineq_constraint_1(x, hessian=False):
    """Constraint: -x - y + 1 ≤ 0 (from y ≥ -x + 1)"""
    g_val = -x[0] - x[1] + 1
    g_grad = np.array([-1.0, -1.0])

    if hessian:
        g_hess = np.zeros((2, 2))
        return g_val, g_grad, g_hess
    return g_val, g_grad


def lp_ineq_constraint_2(x, hessian=False):
    """Constraint: y - 1 ≤ 0 (i.e., y ≤ 1)"""
    g_val = x[1] - 1
    g_grad = np.array([0.0, 1.0])

    if hessian:
        g_hess = np.zeros((2, 2))
        return g_val, g_grad, g_hess
    return g_val, g_grad


def lp_ineq_constraint_3(x, hessian=False):
    """Constraint: x - 2 ≤ 0 (i.e., x ≤ 2)"""
    g_val = x[0] - 2
    g_grad = np.array([1.0, 0.0])

    if hessian:
        g_hess = np.zeros((2, 2))
        return g_val, g_grad, g_hess
    return g_val, g_grad


def lp_ineq_constraint_4(x, hessian=False):
    """Constraint: -y ≤ 0 (i.e., y ≥ 0)"""
    g_val = -x[1]
    g_grad = np.array([0.0, -1.0])

    if hessian:
        g_hess = np.zeros((2, 2))
        return g_val, g_grad, g_hess
    return g_val, g_grad


# LP constraint data
lp_ineq_constraints = [lp_ineq_constraint_1, lp_ineq_constraint_2,
                       lp_ineq_constraint_3, lp_ineq_constraint_4]
