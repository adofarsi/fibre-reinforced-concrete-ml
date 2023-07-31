import os
import argparse
import numpy as np
from typing import Union, Callable
from tqdm.auto import tqdm, trange
from numpy.random import default_rng

from firedrake import *

from physics_driven_ml.utils import get_logger


def random_field(V, N: int = 1, m: int = 1, σ: float = 0.05,
                 tqdm: bool = False, seed: int = 2023):
    """Generate N 2D random fields with m modes."""
    rng = default_rng(seed)
    x, y = SpatialCoordinate(V.ufl_domain())
    fields = []
    for _ in trange(N, disable=not tqdm):
        r = 0
        for _ in range(m):
            # E for concrete 20-40GPa, nu 0.15-0.25
            a, b = rng.uniform(0.15, 0.25, 2)
            k1, k2 = rng.normal(0.2, σ, 2)
            θ = 2 * pi * (k1 * x + k2 * y)
            r += Constant(a) * cos(θ) + Constant(b) * sin(θ)
        ep = sqrt(1 / m) * r

        fields.append(interpolate(ep, V))
    return fields


def generate_data(V, dataset_dir: str, ntrain: int = 50, ntest: int = 10,
                  forward: Union[str, Callable] = "heat", noise: Union[str, Callable] = "normal",
                  scale_noise: float = 1., seed: int = 1234):
    """Generate train/test data for a given PDE-based forward problem and noise distribution.

    Parameters:
        - V: Firedrake function space
        - dataset_dir: directory to save the generated data
        - ntrain: number of training samples
        - ntest: number of test samples
        - forward: forward model (e.g "heat")
        - noise: noise distribution to form the observed data (e.g. "normal")
        - scale_noise: noise scaling factor
        - seed: random seed

    Custom forward problems:
        One can provide a custom forward problem by specifying a callable for the `forward` argument.
        This callable should take in a list of randomly generated inputs and the function space `V`, and
        it should return a list of Firedrake functions corresponding to the PDE solutions.

    Custom noise perturbations:
        Likewise, one can provide a custom noise perturbation by specifying a callable for the `noise` argument.
        This callable should take in a list of PDE solutions, and it should return a list of Firedrake functions
        corresponding to the observed data, i.e. the perturbed PDE solutions.
    """

    logger.info("\n Generate random fields")

    logger.info("\n Generate corresponding PDE solutions")

    if forward == "heat":
        ks = random_field(V, N=ntrain+ntest, tqdm=True, seed=seed)
        us = []
        v = TestFunction(V)
        x, y = SpatialCoordinate(V.ufl_domain())
        f = Function(V).interpolate(sin(pi * x) * sin(pi * y))
        bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
        for k in tqdm(ks):
            u = Function(V)
            F = (inner(exp(k) * grad(u), grad(v)) - inner(f, v)) * dx
            # Solve PDE using LU factorisation
            solve(F == 0, u, bcs=bcs, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
            us.append(u)

    elif forward == "linear_elastic":
        ks = []
        us = []
        V_vec = VectorFunctionSpace(mesh, "CG", 1)  # create a vector function space
        x, y = SpatialCoordinate(V_vec.ufl_domain())

        v, u_ = TestFunction(V_vec), TrialFunction(V_vec)
        u = Function(V_vec, name="Displacement")
        E = Constant(1.0)  # Young's modulus
        nu = Constant(0.2)

        # Lamé parameter
        lmbda = E*nu/(1+nu)/(1-2*nu)
        mu = E/2/(1+nu)
        # Apply a random force as f which like heat source in "heat"
        f_ramdon = as_vector([sin(pi * x) * sin(pi * y), sin(pi * x) * sin(pi * y)])
        f = interpolate(f_ramdon, V_vec)

        # Define strain and stress's formular
        def eps(v):
            return sym(grad(v))
        def sig(v):
            d = 2
            return lmbda*tr(eps(v))*Identity(d) + 2*mu*eps(v)

        # Apply random_field as input strain in x, y, xy direction
        # Instead of store the tensor, I store the strain in each direction
        # Expand the number of random fields to three times original N to ensure that 
        # the number of strain fields generated is consistent with N.
        fields = random_field(V=V, N = 3*(ntrain+ntest), tqdm = True, seed=seed)
        for a in range(0, len(fields)-2, 3):  # process the fields three by three
            exx = interpolate(fields[a], V)
            eyy = interpolate(fields[a+1], V)
            exy = interpolate(fields[a+2], V)

            # Save the strain fields
            epsilon = []
            epsilon.append(exx)
            epsilon.append(eyy)
            epsilon.append(exy)
            ks.append(epsilon)

            # Define the displacement boundary conditions
            bc_expr = np.array([exx + exy * x, eyy + exy * y])
            # Boundary conditions
            bcL = DirichletBC(V_vec, bc_expr, 1)
            bcR = DirichletBC(V_vec, bc_expr, 2)
            bcB = DirichletBC(V_vec, bc_expr, 3)
            bcT = DirichletBC(V_vec, bc_expr, 4)
            # Define variational problem using the principle of virtual work
            a = inner(sig(u_), eps(v)) * dx
            L = inner(f, v) * dx

            # Solve PDE
            solve(a == L, u, bcs=[bcL, bcB, bcR, bcT], solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

            # Use the displacement to calculate the stress in each direction
            sxx = sig(u)[0, 0]
            syy = sig(u)[1, 1]
            sxy = sig(u)[0, 1]

            # Save the stress fields
            sigma = []
            sigma.append(sxx)
            sigma.append(syy)
            sigma.append(sxy)

            us.append(sigma)

    elif callable(forward):
        us = forward(ks, V)
    else:
        raise NotImplementedError("Forward problem not implemented. Use 'heat' or provide a callable for your forward problem.")

    logger.info("\n Form noisy observations from PDE solutions")

    if noise == "normal":
        us_obs = []
        for u in tqdm(us):
            u_obs = Function(V).assign(u)
            noise = scale_noise * np.random.rand(V.dim())
            # Add noise to PDE solutions
            u_obs.dat.data[:] += noise
            us_obs.append(u_obs)
    elif noise == "linear_elastic":
        us_obs = []
        # V_ten = TensorFunctionSpace(mesh, "CG", 1)
        noise = scale_noise * np.random.rand(V.dim())
        # Add noise to PDE solutions
        # For us is a list of list, so we need to add noise to each element in the list
        for u in tqdm(us):
            for _ in range(3):  # process the fields three by three
                nxx = interpolate(u[0], V)
                nyy = interpolate(u[1], V)
                nxy = interpolate(u[2], V)

                nxx.dat.data[:] += noise
                nyy.dat.data[:] += noise
                nxy.dat.data[:] += noise

                add_u_obs = []
                add_u_obs.append(nxx)
                add_u_obs.append(nyy)
                add_u_obs.append(nxy)

                us_obs.append(add_u_obs)
                
    elif callable(noise):
        us_obs = noise(us)
    else:
        raise NotImplementedError("Noise distribution not implemented. Use 'normal' or provide a callable for your noise distribution.")

    logger.info(f"\n Generated {ntrain} training samples and {ntest} test samples.")

    # Split into train/test
    ks_train, ks_test = ks[:ntrain], ks[ntrain:]
    us_train, us_test = us[:ntrain], us[ntrain:]
    us_obs_train, us_obs_test = us_obs[:ntrain], us_obs[ntrain:]

    logger.info(f"\n Saving train/test data to {os.path.abspath(dataset_dir)}.")

    # Save train data
    with CheckpointFile(os.path.join(dataset_dir, "train_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = ntrain
        afile.save_mesh(mesh)
        for i, (k, u, u_obs) in enumerate(zip(ks_train, us_train, us_obs_train)):
            afile.save_function(k[0], idx=i, name="k_xx")
            afile.save_function(k[1], idx=i, name="k_yy")
            afile.save_function(k[2], idx=i, name="k_xy")
            afile.save_function(u_obs[0], idx=i, name="u_obs_xx")
            afile.save_function(u_obs[1], idx=i, name="u_obs_yy")
            afile.save_function(u_obs[2], idx=i, name="u_obs_xy")

    # Save test data
    with CheckpointFile(os.path.join(dataset_dir, "test_data.h5"), "w") as afile:
        afile.h5pyfile["n"] = ntest
        afile.save_mesh(mesh)
        # I have three fields as input and three as output, so save them separately
        for i, (k, u, u_obs) in enumerate(zip(ks_test, us_test, us_obs_test)):
            afile.save_function(k[0], idx=i, name="k_xx")
            afile.save_function(k[1], idx=i, name="k_yy")
            afile.save_function(k[2], idx=i, name="k_xy")
            afile.save_function(u_obs[0], idx=i, name="u_obs_xx")
            afile.save_function(u_obs[1], idx=i, name="u_obs_yy")
            afile.save_function(u_obs[2], idx=i, name="u_obs_xy")

if __name__ == "__main__":
    logger = get_logger("Data generation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", default=100, type=int, help="Number of training samples")
    parser.add_argument("--ntest", default=10, type=int, help="Number of test samples")
    parser.add_argument("--forward", default="linear_elastic", type=str, help="Forward problem (e.g. 'heat')")
    parser.add_argument("--noise", default="linear_elastic", type=str, help="Noise distribution (e.g. 'normal')")
    parser.add_argument("--scale_noise", default=5e-3, type=float, help="Noise scaling")
    parser.add_argument("--nx", default=50, type=int, help="Number of cells in x-direction")
    parser.add_argument("--ny", default=50, type=int, help="Number of cells in y-direction")
    parser.add_argument("--Lx", default=1., type=float, help="Length of the domain")
    parser.add_argument("--Ly", default=1., type=float, help="Width of the domain")
    parser.add_argument("--degree", default=1, type=int, help="Degree of the finite element CG space")
    parser.add_argument("--data_dir", default=os.environ["DATA_DIR"], type=str, help="Data directory")
    parser.add_argument("--dataset_name", default="test1", type=str, help="Dataset name")

    args = parser.parse_args()

    # Set up mesh and finite element space
    mesh = RectangleMesh(args.nx, args.ny, args.Lx, args.Ly, name="mesh")
    V = FunctionSpace(mesh, "CG", args.degree)

    # Set up data directory
    dataset_dir = os.path.join(args.data_dir, "datasets", args.dataset_name)
    # Make data directory while dealing with parallelism
    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        # Another process created the directory
        pass
    # Generate data
    generate_data(V, dataset_dir=dataset_dir, ntrain=args.ntrain,
                  ntest=args.ntest, forward=args.forward,
                  noise=args.noise, scale_noise=args.scale_noise)