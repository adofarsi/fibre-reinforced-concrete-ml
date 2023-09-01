from firedrake import *
from tqdm.auto import tqdm
import numpy as np


def forward_model(E, nu, strain_tensor):
    mesh = Mesh("withgap.msh")
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    # Test and trial function for the displacement field
    v = TestFunction(V), TrialFunction(V)
    u = Function(V, name="Displacement")

    # # Plot the mesh
    fig1, ax1 = plt.subplots(figsize=(14, 3))
    triplot(mesh, axes=ax1)
    ax1.set_title("Mesh Plot")

    # Show the plot
    plt.show()

    # Lamé parameter
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)

    # Constitutive equations
    def eps(v):
        return 0.5*(grad(v) + grad(v).T)

    def sigma(v):
        d = 2
        return lmbda*tr(eps(v))*Identity(d) + 2*mu*eps(v)
    # Body force
    # f = Constant((0.0, 0.0))
    # Facet normal vector in each boundary
    n = FacetNormal(mesh)

    # Convert strain tensor into displacement boundary conditions
    eyy = strain_tensor[1, 1]

    # boundary conditions
    punch_u = -eyy  # total imposed vertical displacement at the top loading point
    # loading by displacement control
    punch_bc = (punch_u-u[1])*dot(v, n)*ds(3)
    # weak form of the vertical displacement constraints at the bottom supports
    left_bc = (0-u[1])*dot(v, -n)*ds(1) 
    right_bc = (0-u[1])*dot(v, -n)*ds(2)

    F_ext = punch_bc + left_bc + right_bc   # external virtual work
    F = inner(sigma(u), eps(v))*dx + F_ext  # residual form of the variational problem

    # Solve PDE
    solve(F == 0, u, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    # print(u.dat.data)

    fig1, axes = plt.subplots(figsize=(14, 3))
    axes.set_aspect('equal')
    contours = tricontourf(u, levels=50, axes=axes)
    plt.colorbar(contours)
    plt.title("Value of Displacement Field(u)")
    plt.figure(figsize=(14, 3))
    plt.show()

    # Get the coordinates of the mesh nodes
    coords = mesh.coordinates.dat.data

    # Extract the x and y displacements
    u_values = u.dat.data
    u_x = u_values[:, 0]
    u_y = u_values[:, 1]

    # Create a quiver plot
    plt.figure(figsize=(14, 3))
    plt.quiver(coords[:, 0], coords[:, 1], u_x, u_y, scale=20, width=0.002)
    plt.title("Displacement Field(u)")
    plt.show()

    displaced_coordinates = interpolate(SpatialCoordinate(mesh) + u, V)
    displaced_mesh = Mesh(displaced_coordinates)
    # # NBVAL_IGNORE_OUTPUT
    fig, axes = plt.subplots(figsize=(14, 3))
    axes.set_title("Displaced Mesh")

    triplot(displaced_mesh, axes=axes)
    axes.set_aspect("equal")

    W = TensorFunctionSpace(mesh, 'CG', 1)  # Define a tensor function space
    projected_sigma = project(sigma(u), W)

    sig = Function(W)  # Define sig as a Function over the tensor function space
    sig.assign(projected_sigma)  # Assign the projected stress to sig

    return sig


# fire version
def get_dataset(ntrain, ntest, path):
    X, y = [], []

    for i in tqdm(range(ntrain + ntest)):
        # Randomly generate E and nu within given ranges
        E = np.random.uniform(30e3, 90e3)  # Young's modulus in Pa
        nu = np.random.uniform(0.1, 0.3)      # Poisson's ratio

        # Generate diagonal elements
        a11, a12, a22 = [np.random.uniform(-0.1, 0.1) for _ in range(3)]
        # Construct the 2x2 matrix
        strain = np.array([[a11, a12], [a12, a22]])
        stress = check_forward(E, nu, strain)

        # Flatten and concatenate [E, nu] and strain
        input_data = np.hstack([E, nu, a11, a22, a12])

        X.append(input_data)
        y.append([stress[0, 0], stress[1, 1], stress[0, 1]])
        # y.append([stress.dat.data[0,0,0], stress.dat.data[0,1,1], stress.dat.data[0,0,1]])
    # print(y)
    # Convert lists to numpy arrays
    X_train, X_test = np.array(X[:ntrain]), np.array(X[ntrain:])
    y_train, y_test = np.array(y[:ntrain]), np.array(y[ntrain:])

    np.save(f"{path}/X_train1.npy", X_train)
    np.save(f"{path}/y_train1.npy", y_train)
    np.save(f"{path}/X_test1.npy", X_test)
    np.save(f"{path}/y_test1.npy", y_test)


if __name__ == "__main__":
    ntrain = 4000
    ntest = 400
    path = "../../data/datasets/linear_three_point"
    get_dataset(ntrain, ntest, path)
