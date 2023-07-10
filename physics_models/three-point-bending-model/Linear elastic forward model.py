import numpy as np
import pandas as pd
from firedrake import *

# Define function to calculate stress tensor from strain tensor
def linear_elasticity_model(E, nu, strain):
    # Convert input parameters to Firedrake Constants
    E = Constant(E)
    nu = Constant(nu)

    # Calculate Lame's constants
    mu = E / (2*(1 + nu))
    lmbda = E*nu / ((1 + nu)*(1 - 2*nu))

    # Calculate stress tensor
    sigma = 2*mu*strain + lmbda*tr(strain)*Identity(2)

    return sigma


def get_dataset(num_samples, E, nu):
    mesh = UnitSquareMesh(10, 10)
    V = TensorFunctionSpace(mesh, "CG", 1)
    Vc = TensorFunctionSpace(mesh, "CG", 1)
    X, y = [], []
    for _ in range(num_samples):
        epsilon = Function(V)
        strain = np.random.rand(2, 2)
        epsilon.interpolate(as_tensor(strain))
        sigma = linear_elasticity_model(E, nu, epsilon)
        sigma_proj = project(sigma, Vc)
        X.append(strain.reshape(-1))
        y.append(sigma_proj.dat.data[0, :, :].reshape(-1))
    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == '__main__':
    # Generate a dataset
    num_samples = 5
    E = Constant(2.1e11)  # Young's modulus in Pa
    nu = Constant(0.3)  # Poisson's ratio
    X, y = get_dataset(num_samples, E, nu)
    # Store data in a CSV file
    data = pd.DataFrame({"strain": X, "stress": y})
    data.to_csv("/Users/mh522/Documents/new/graduation design/6.28code/three_point_bending/linear_data.csv", index=False)

