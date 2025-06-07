"""
Utility functions for visualization and common operations.
Following HW01 structure requirements.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d_simplex_with_path(path, obj_vals, x_final,
                              title="3D Feasible Region and Path"):
    """
    Create 3D plot showing feasible simplex and optimization path.

    Args:
        path: List of points along optimization path
        obj_vals: Objective values at each iteration
        x_final: Final solution point
        title: Plot title

    Returns:
        fig: matplotlib figure object
    """
    path_array = np.array(path)

    fig = plt.figure(figsize=(15, 6))

    # Plot 1: 3D feasible region and central path
    ax1 = fig.add_subplot(121, projection='3d')

    # Define simplex vertices (x+y+z=1, x,y,z≥0)
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Plot simplex edges
    edges = [[0, 1], [1, 2], [2, 0]]
    for edge in edges:
        pts = vertices[edge]
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', alpha=0.6, linewidth=2)

    # Plot feasible region (triangle)
    triangle = [vertices]
    ax1.add_collection3d(Poly3DCollection(
        triangle, alpha=0.2, facecolor='lightblue', edgecolor='blue'))

    # Plot central path
    ax1.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'ro-',
             markersize=4, linewidth=2, alpha=0.8, label='Central Path')
    ax1.plot(path_array[0, 0], path_array[0, 1], path_array[0, 2], 'go',
             markersize=8, label='Initial Point')
    ax1.plot(x_final[0], x_final[1], x_final[2], 'rs',
             markersize=10, label='Final Solution')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(title)
    ax1.legend()

    # Plot 2: Convergence
    ax2 = fig.add_subplot(122)
    ax2.plot(range(len(obj_vals)), obj_vals, 'b-o', markersize=5, linewidth=2)
    ax2.set_xlabel('Outer Iteration')
    ax2.set_ylabel('Objective Value')
    ax2.set_title('Convergence')
    ax2.grid(True, alpha=0.3)

    return fig


def plot_2d_feasible_region_with_path(path, obj_vals, x_final,
                                      title="2D Feasible Region and Path",
                                      convert_to_max=True):
    """
    Create 2D plot showing feasible region and optimization path.

    Args:
        path: List of points along optimization path
        obj_vals: Objective values at each iteration
        x_final: Final solution point
        title: Plot title
        convert_to_max: Whether to convert minimization to maximization
    Returns:
        fig: matplotlib figure object
    """
    path_array = np.array(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x_range = np.linspace(-0.1, 2.2, 100)
    y1 = -x_range + 1
    vertices_x = [0, 2, 1, 0]
    vertices_y = [1, 1, 0, 1]

    ax1.fill(vertices_x, vertices_y, alpha=0.3,
             color='lightblue', label='Feasible Region')

    # Plot constraint boundaries
    ax1.plot(x_range, y1, 'b--', label='y = -x + 1', alpha=0.7)
    ax1.axhline(y=1, color='r', linestyle='--', label='y = 1', alpha=0.7)
    ax1.axvline(x=2, color='g', linestyle='--', label='x = 2', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot central path
    ax1.plot(path_array[:, 0], path_array[:, 1], 'ro-',
             markersize=4, linewidth=2, alpha=0.8, label='Central Path')
    ax1.plot(path_array[0, 0], path_array[0, 1], 'go',
             markersize=8, label='Initial Point')
    ax1.plot(x_final[0], x_final[1], 'rs',
             markersize=10, label='Final Solution')

    ax1.set_xlim(-0.1, 2.2)
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Objective value vs iteration
    display_vals = [-obj for obj in obj_vals] if convert_to_max else obj_vals
    ylabel = 'Objective Value (x + y)' if convert_to_max else 'Objective Value'

    ax2.plot(range(len(display_vals)), display_vals,
             'b-o', markersize=5, linewidth=2)
    ax2.set_xlabel('Outer Iteration')
    ax2.set_ylabel(ylabel)
    ax2.set_title('Convergence')
    ax2.grid(True, alpha=0.3)

    return fig


def save_and_show_plot(fig, filename, show=True):
    """
    Save plot to file and optionally display it.

    Args:
        fig: matplotlib figure object
        filename: Output filename
        show: Whether to display the plot
    """
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def print_optimization_results(problem_name, x_final, obj_final,
                               constraints=None):
    """
    Print formatted optimization results.

    Args:
        problem_name: Name of the optimization problem
        x_final: Final solution point
        obj_final: Final objective value
        constraints: Optional dict of constraint values to print
    """
    print(f"\n=== {problem_name} Results ===")
    print(f"Final solution: x={x_final}")
    print(f"Final objective value: {obj_final:.6f}")

    if constraints:
        print("Constraint values:")
        for name, value in constraints.items():
            print(f"  {name}: {value:.6f}")
