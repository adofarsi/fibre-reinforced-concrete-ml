import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import pandas as pd

def three_point_bending(E, nu, w_max):
    # Create mesh and function space
    length = 1.0  # Length of the beam
    b = 0.05  # Thickness of the beam
    h = 0.1   # Height of the beam
    n = 100  # Number of elements

    mesh = IntervalMesh(n, length)

    V = FunctionSpace(mesh, "CG", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define Dirichlet boundary conditions
    left_bc = DirichletBC(V, 0.0, 1)
    right_bc = DirichletBC(V, 0.0, 2)
    bc = [left_bc, right_bc]

    # Calculate moment of inertia
    I = b * h**3 / 12

    # distributed load
    f = Constant(-1.0)
    F = E*I*u.dx(0).dx(0)*v*dx - f*v*dx

    # Solve the problem
    w = Function(V)
    solve(lhs(F) == rhs(F), w, bcs=bc)

    # Compute the deflection
    force = 48 * w_max * E * I / length**3

    return force

def get_dataset(num_samples, E, nu):
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    Vc = FunctionSpace(mesh, "CG", 1)
    X, y = [], []
    for _ in range(num_samples):
        w = np.random.rand()
        w_max = Function(V).interpolate(Constant(w))
        force = three_point_bending(E, nu, w_max)
    #     force_proj = interpolate(force, Vc)
    #     X.append(w_max.vector().get_local()[0])
    #     y.append(force_proj.vector().get_local()[0])
    #     # Store data in a CSV file
    #     data = pd.DataFrame({"w_max": X, "force": y})
    #     data.to_csv("/Users/mh522/Documents/new/graduation design/6.28code/three_point_bending/data.csv", index=False)
    # return X, y

if __name__ == '__main__':
    # Generate a dataset
    num_samples = 500
    E = Constant(2.1e3)  # Young's modulus in GPa
    nu = Constant(0.3)  # Poisson's ratio
    X, y = get_dataset(num_samples, E, nu)
    
    # Plot the force-deflection curve
    plt.plot(X, y, 'ro')
    plt.xlabel("Deflection")
    plt.ylabel("Force")
    plt.title("Three-Point Bending Test")
    plt.show()
