# Numerical Optimization Programming Ex02

## Implementation Summary

This project implements an interior point method for solving constrained optimization problems using the log-barrier approach.

### Components Implemented

1. **`src/constrained_min.py`** - Main interior point method implementation
   - `interior_pt()` function using log-barrier method
   - Newton's method for equality-constrained subproblems
   - Automatic feasibility projection for boundary cases
   - Parameters: t=1 initial, μ=10 multiplier, tolerance=1e-8

2. **`src/unconstrained_min.py`** - Supporting optimization utilities
   - Newton's method implementation
   - Backtracking line search with Armijo condition

3. **`examples.py`** - Problem definitions
   - QP problem: minimize x² + y² + (z+1)² subject to x+y+z=1, x,y,z≥0
   - LP problem: maximize x + y subject to y≥-x+1, y≤1, x≤2, y≥0

4. **`tests/test_constrained_min.py`** - Unit tests with visualization
   - `test_qp()` and `test_lp()` functions
   - Generates 3D plots for QP showing feasible simplex and central path
   - Generates 2D plots for LP showing feasible region and solution path

### Testing

Run tests with:
```bash
python -m unittest tests.test_constrained_min -v
```

This generates plots saved as `qp_results.png` and `lp_results.png`.
