import numpy as np


class Function:
    """Base class for optimization functions."""

    def __init__(self, hessian_needed: bool = True):
        self.hessian_needed = hessian_needed

    def objective(self, x: np.ndarray) -> float:
        """Compute objective function value."""
        raise NotImplementedError("Subclasses must implement objective()")

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient vector."""
        raise NotImplementedError("Subclasses must implement gradient()")

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix."""
        raise NotImplementedError("Subclasses must implement hessian()")


class QuadraticFunction(Function):
    """Generic quadratic function: f(x) = x^T @ Q @ x + c^T @ x + bias"""

    def __init__(self, Q: np.ndarray, c: np.ndarray = None, bias: float = 0.0):
        super().__init__(hessian_needed=True)
        self.Q = Q
        self.c = c if c is not None else np.zeros(Q.shape[0])
        self.bias = bias

    def objective(self, x: np.ndarray) -> float:
        return x @ self.Q @ x + self.c @ x + self.bias

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.Q @ x + self.c

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.Q


class LinearFunction(Function):
    """Generic linear function: f(x) = c^T @ x + bias"""

    def __init__(self, c: np.ndarray, bias: float = 0.0):
        super().__init__(hessian_needed=True)
        self.c = c
        self.bias = bias

    def objective(self, x: np.ndarray) -> float:
        return self.c @ x + self.bias

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.c

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((len(self.c), len(self.c)))


class LinearConstraint(Function):
    """Generic linear constraint: a^T @ x + b ≤ 0"""

    def __init__(self, a: np.ndarray, b: float = 0.0, name: str = ""):
        super().__init__(hessian_needed=True)
        self.a = a
        self.b = b
        self.name = name

    def objective(self, x: np.ndarray) -> float:
        return self.a @ x + self.b

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.a

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((len(self.a), len(self.a)))


class QPObjective(QuadraticFunction):
    """QP Objective: min x² + y² + (z+1)²"""

    def __init__(self):
        # x² + y² + (z+1)² = x² + y² + z² + 2z + 1
        Q = np.eye(3)  # Coefficients for x², y², z²
        c = np.array([0.0, 0.0, 2.0])  # Linear term: 2z
        bias = 1.0  # Constant term
        super().__init__(Q=Q, c=c, bias=bias)


class LPObjective(LinearFunction):
    """LP Objective: max x + y converted to min -(x + y)"""

    def __init__(self):
        c = np.array([-1.0, -1.0])  # Coefficients for maximizing x + y
        super().__init__(c=c, bias=0.0)


# Create constraint instances manually (simpler approach)
qp_ineq_constraints = [
    LinearConstraint(np.array([-1.0, 0.0, 0.0]), 0.0, "x >= 0"),
    LinearConstraint(np.array([0.0, -1.0, 0.0]), 0.0, "y >= 0"),
    LinearConstraint(np.array([0.0, 0.0, -1.0]), 0.0, "z >= 0"),
]

# LP constraints using manual creation for specific constraints
lp_ineq_constraints = [
    LinearConstraint(np.array([-1.0, -1.0]), 1.0, "x + y >= 1"),
    LinearConstraint(np.array([0.0, 1.0]), -1.0, "y <= 1"),
    LinearConstraint(np.array([1.0, 0.0]), -2.0, "x <= 2"),
    LinearConstraint(np.array([0.0, -1.0]), 0.0, "y >= 0"),
]

# Create function instances
qp_objective = QPObjective()
lp_objective = LPObjective()

# Constraint matrices for QP
qp_eq_constraints_mat = np.array([1.0, 1.0, 1.0])
qp_eq_constraints_rhs = 1.0
