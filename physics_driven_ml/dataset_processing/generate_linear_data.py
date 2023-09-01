from firedrake import *
import numpy as np


def forward_model(E, nu, strain_tensor):
    # Mesh refinement parameters
    nx, ny = 20, 20
    mesh = RectangleMesh(nx, ny, 1, 1)
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    # Test and trial function for the displacement field
    v, u_ = TestFunction(V), TrialFunction(V)
    u = Function(V, name="Displacement")

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
    f = Constant((0.0, 0.0))

    # Convert strain tensor into displacement boundary conditions
    exx, eyy, exy = strain_tensor[0, 0], strain_tensor[1, 1], strain_tensor[0, 1]
    uLx = - exx/2 + exy * (y-.5)
    uLy = eyy * (y-.5) - exy/2
    uRx = exx/2 + exy * (y-.5)
    uRy = eyy * (y-.5) + exy/2

    uBx = exx * (x-.5) - exy/2
    uBy = - eyy/2 + exy * (x-.5)
    uTx = exx * (x-.5) + exy/2
    uTy = eyy/2 + exy * (x-.5)

    # Boundary conditions
    bcL = DirichletBC(V, [uLx, uLy], 1)
    bcR = DirichletBC(V, [uRx, uRy], 2)
    bcB = DirichletBC(V, [uBx, uBy], 3)
    bcT = DirichletBC(V, [uTx, uTy], 4)

    # Formal equation is div(sigma(u)) = f
    # Form
    a = inner(sigma(u_), eps(v)) * dx
    L = inner(f, v) * dx
    # Solve PDE
    solve(a == L, u, bcs=[bcL, bcB, bcR, bcT], solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    # Boundary stress
    sxx = assemble(.5*sigma(u)[0, 0] * ds(1) + .5*sigma(u)[0, 0] * ds(2))
    syy = assemble(.5*sigma(u)[1, 1] * ds(3) + .5*sigma(u)[1, 1] * ds(4))
    sxy_syx = assemble(.25*sigma(u)[0, 1] * ds(1) +
                       .25*sigma(u)[0, 1] * ds(2) +
                       .25*sigma(u)[1, 0] * ds(3) +
                       .25*sigma(u)[1, 0] * ds(4))
    # Enforce symmetry: Sxy = Syx

    stress_tensor = np.array([[sxx, sxy_syx],
                              [sxy_syx, syy]])

    return stress_tensor


# check
def check_forward(E, nu, strain_tensor):
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    # lmbda*tr(eps(v))*Identity(d) + 2*mu*eps(v), 0.1 is the trace of eps(v)
    s = lmbda*np.trace(strain_tensor)*np.eye(2) + 2*mu*strain_tensor
    return s


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
    path = "../../data/datasets/linear_data"
    get_dataset(ntrain, ntest, path)
