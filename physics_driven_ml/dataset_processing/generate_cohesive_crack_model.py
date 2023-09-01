from firedrake import *
from IPython.display import clear_output
import csv


def forward_model(E, nu): 
    # beam geometry: length, height = 700, 150 [mm]
    # load the mesh generated with Gmsh
    mesh = Mesh('withgap.msh')

    n = FacetNormal(mesh)
    # n('-')`: This notation is used in the context of discontinuous Galerkin methods or when dealing with interior facets
    # The `perp` function computes the perpendicular (or tangent) to the given vector
    n_, t_ = n('-'), perp(n('-'))

    # define FE spaces and functions
    V = VectorFunctionSpace(mesh, "DG", 1)          # space for discontinuous displacement vector (u)
    W = VectorFunctionSpace(mesh, "DG", 1, dim=3)   # space for discontinuous stress tensor (sig)
    S = FunctionSpace(mesh, "HDiv Trace", 0)        # space for facet-based internal variables (dmax_n, dmax_t, gmax, flag)
    u = Function(V, name="Displacement [mm]")
    v = TestFunction(V), TrialFunction(V)
    sig = Function(W, name="Stress [MPa]")
    dmax_n = Function(S, name="Maximum normal opening [mm]")
    dmax_t = Function(S, name="Maximum tangential opening [mm]")
    gmax = Function(S, name="Maximum effective opening [-]")
    flag = Function(S, name="Joint state flag")     # (1: active; 0: failed)

    # create rigid body modes
    x, y = SpatialCoordinate(mesh)
    b0 = Function(V)
    b1 = Function(V)
    b2 = Function(V)
    b0.interpolate(Constant([1, 0]))
    b1.interpolate(Constant([0, 1]))
    b2.interpolate(as_vector([-y, x]))
    nullmodes = VectorSpaceBasis([b0, b1, b2])
    nullmodes.orthonormalize()

    dS_measure = dS(degree=2)   # interior facet integration measure

    def facet_project(v, V):

        v_, dv = TestFunction(V), TrialFunction(V)
        a = inner(avg(v_), avg(dv))*dS_measure
        if isinstance(v, dict):
            L = sum([inner(avg(v_), val)*dS_measure(reg) for (reg, val) in v.items()])
        else:
            L = inner(avg(v_), v)*dS_measure
        u = Function(V)
        solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi'})
        return u

    # elastic parameters
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    # constitutive parameters
    sig0 = Constant(3)              # uniaxial tensile strength [MPa]
    c = Constant(3)                 # internal cohesion [MPa]
    phi = Constant(np.deg2rad(35))  # internal friction angle [degree]
    pen = 0.1*E                     # penalty stiffness

    # strain tensor
    def epsilon(u_):
        return sym(grad(u_))        # = 0.5*(grad(u_) + grad(u_).T)
    # elastic stress -strain relation
    # def sigma(u_):
    #     I = Identity(u_.geometric_dimension()) # identity tensor
    #     return lmbda*tr(epsilon(u_))*I + 2*mu*epsilon(u_)
 
    def sigma(u_):
        dim = mesh.geometric_dimension()  # Get geometric dimension from the mesh
        I = Identity(dim)  # identity tensor
        return lmbda*tr(epsilon(u_))*I + 2*mu*epsilon(u_)

    # normal opening displacement
    def d_n(u_):
        return dot(jump(u_), n_)
    # tangential opening displacement

    def d_t(u_):
        return dot(jump(u_), t_)

    # define traction -separation laws
    pos = lambda X: (X+abs(X))/2     # take the positive part of X 
    Gf1 = Constant(36e-3)            # Mode I initial fracture energy [N/mm]
    GF1 = 75*Gf1                     # Mode I total fracture energy
    Gf2, GF2 = 4*Gf1, 4*GF1          # Mode II initial/total fracture energy
    # parameters for mode I traction-separation law in the normal direction 
    f0_n = sig0                      # peak traction
    d0_n = f0_n/pen                  # normal opening displacement corresponding to the peak traction
    dk_n = Constant(0.016)           # normal opening displacement corresponding to the kink point [mm]
    k = 1-dk_n*f0_n/2/Gf1
    fk_n = k*f0_n                    # normal traction corresponding to the kink point 
    df_n = 2*(GF1-(1-k)*Gf1)/fk_n    # normal opening displacement corresponding to null traction
    s1_n = (fk_n-f0_n)/(dk_n-d0_n)   # slope of the first softening branch 
    s2_n = -fk_n/(df_n-dk_n)         # slope of the second softening branch
    # parameters for mode II traction-separation law in the tangential direction
    f0_t = c                         # peak traction (without frictional contribution)
    d0_t = f0_t/pen                  # tangential opening displacement corresponding to the peak traction
    fk_t = k*f0_t                    # tangential traction corresponding to the kink point 
    dk_t = 2*Gf2*(1-k)/f0_t          # tangential opening displacement corresponding to the kink point
    df_t = 2*(GF2-(1-k)*Gf2)/fk_t    # tangential opening displacement corresponding to null traction
    s1_t = (fk_t-f0_t)/(dk_t-d0_t)   # slope of the first softening branch 
    s2_t = -fk_t/(df_t-dk_t)         # slope of the second softening branch

    # effective opening
    def g(u_):
        return sqrt((pos(d_n(u_))/df_n)**2 + (d_t(u_)/df_t)**2)
    # virgin -loading definition of the traction -separation laws

    def Tn_virgin(d_n):
        return conditional(d_n <= d0_n, pen*d_n, conditional(d_n <= dk_n, f0_n + s1_n*(d_n-d0_n), fk_n + s2_n*(d_n-dk_n)))

    def Tt_virgin(d_t):
        return conditional(abs(d_t) <= d0_t, pen*d_t, sign(d_t)*(conditional(abs(d_t) <= dk_t, f0_t + s1_t*(abs(d_t)-d0_t), fk_t + s2_t*(abs(d_t)-dk_t)) + pen * pos(-d_n(u)) * tan(phi)))
    # unloading/reloding definition of the traction-separation laws

    def Tn_ur(d_n, dmax):
        return conditional(dmax <= d0_n, pen*d_n, Tn_virgin(dmax)/dmax * d_n)

    def Tt_ur(d_t, dmax):
        return conditional(dmax <= d0_t, pen*d_t, Tt_virgin(dmax)/dmax * d_t)  # complete definition of the traction-separation laws

    def Tn(d_n, dmax):
        return conditional(d_n >= dmax, Tn_virgin(d_n), Tn_ur(d_n, dmax))

    def Tt(d_t, dmax):
        return conditional(abs(d_t) >= dmax, Tt_virgin(d_t), Tt_ur(d_t, dmax))

    flag.assign(facet_project(1, S))  # initial activeness of all the jointelements along the interior facets
    T = (Tn(d_n(u), avg(dmax_n))*n_ + Tt(d_t(u), avg(dmax_t))*t_)*avg(flag)
    # total traction vector

    # boundary conditions
    m = Constant(0)  # loading multiplier
    punch_u = -0.2   # total imposed vertical displacement at the top loading point
    # loading by displacement control
    punch_bc = pen/20*(m*punch_u - u[1]) * dot(v, n)*ds(3)
    # weak form of the vertical displacement constraints at the bottom supports
    left_bc = pen/20*(0-u[1])*dot(v, -n)*ds(1)
    right_bc = pen/20*(0-u[1])*dot(v, -n)*ds(2)
    F_ext = punch_bc + left_bc + right_bc  # external virtual work
    F = inner(sigma(u), epsilon(v))*dx + inner(T, jump(v))*dS - F_ext  # residual form of the variational problem
    # define the nonlinear problem and set the solver
    problem = NonlinearVariationalProblem(F, u)
    solver_parameters = {"ksp_type": "fgmres", "pc_type": "ilu", 'snes_atol': 1e-5}
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, nullspace=nullmodes)

    # represent a 2D tensor as a 3-dimensional vector
    def vector_2D(X):
        return as_vector([X[0, 0], X[1, 1], X[0, 1]])
    # create output file
    outfile = File("train_data_e25nu02/3pb_czm_pc.pvd")
    outfile.write(u, sig, time=0)
    freq = 100
    # load-stepping loop
    steps = np.concatenate((np.linspace(0, 1.2, 2000), np.linspace(1.201, 2, 9000)))

    # np.linspace(3, 3.3, 1000)))
    # steps = np.concatenate((np.linspace(0, 0.9, 100), np.linspace(0.9, 1.2, 2000)))
    sigx = [0]  # horizontal stress at notch tip [MPa] x
    CMOD = [0]  # crack mouth opening displacement [mm]
    RF = [0]    # vertical reaction force at top loading point [N/mm] P
    disp = [0]  # vertical displacement at top loading point [mm] U
    EE = [0]    # Young's modulus [MPa] E
    NU = [0]    # Poisson ratio [-] nu
    # Test data
    sigx_t = [0]  # horizontal stress at notch tip [MPa] x
    CMOD_t = [0]  # crack mouth opening displacement [mm]
    RF_t = [0]    # vertical reaction force at top loading point [N/mm] P
    disp_t = [0]  # vertical displacement at top loading point [mm] U
    EE_t = [0]    # Young's modulus [MPa] E
    NU_t = [0]    # Poisson ratio [-] nu

    for (i, t) in enumerate(steps[1:]):
        print("Increment:", i+1)
        m.assign(t)     # increase the loading increment by increment
        solver.solve()  # solve the nonlinear equation for u
        # once convergent solution is obtained for the current load increment , update stress and internal variables
        sig.assign(project(vector_2D(sigma(u)), W))
        dn_, dt_, g_ = facet_project(d_n(u), S), facet_project(d_t(u), S), facet_project(g(u), S)

        dmax_n_proj = project(conditional(dn_ > dmax_n, dn_, dmax_n), S)
        dmax_n.assign(dmax_n_proj)

        dmax_t_proj = project(conditional(abs(dt_) > dmax_t, abs(dt_), dmax_t), S)
        dmax_t.assign(dmax_t_proj)
        gmax_proj = project(conditional(g_ > gmax, g_, gmax), S)
        gmax.assign(gmax_proj)

        # add sigma to combine the result from linear
        # sig_proj = project(sigma(u), L)
        # sigma_count.assign(sig_proj)

        flag.assign(facet_project(conditional(avg(gmax) < 1, 1, 0), S))
        # update and print the residual
        # res = norm(assemble(F))
        # print(" Residual:", res)
        clear_output(wait=True)
        #
        if i % 10 == 0:
            sigx.append(sig([350, 50])[0])
            CMOD.append(u([352.5, 0])[0]-u([347.499, 0])[0])
            RF.append(assemble(sig[1]*ds(3)))
            disp.append(u([350, 149.999])[1])
            EE.append(E.dat.data[0])
            NU.append(nu.dat.data[0])
            # sigma_xx.append(sig([350, 50])[0,0])
            # sigma_xy.append(sig([350, 50])[0,1])
            # sigma_yy.append(sig([350, 50])[1,1])

        # Test data

        if i % 14 == 0:
            if i % 10 != 0:
                sigx_t.append(sig([350, 50])[0])
                CMOD_t.append(u([352.5, 0])[0]-u([347.499, 0])[0])
                RF_t.append(assemble(sig[1] * ds(3)))
                disp_t.append(u([350, 149.999])[1])
                EE_t.append(E.dat.data[0])
                NU_t.append(nu.dat.data[0])
        #
        if i % freq == 0:
            outfile.write(u, sig, time=t)

    # Prepare data
    data = list(zip(sigx, RF, CMOD, disp, steps, EE, NU))
    data_test = list(zip(sigx_t, RF_t, CMOD_t, disp_t, steps, EE_t, NU_t))

    # Write to CSV
    with open('../../data/datasets/cohesive_crack_data/cohesive_crack_train.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sigx", "RF", "CMOD", "disp", "steps", "E", "nu"])  # Header
        writer.writerows(data)  # Data

    with open('../../data/datasets/cohesive_crack_data/cohesive_crack_test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sigx", "RF", "CMOD", "disp", "steps", "E", "nu"])  # Header
        writer.writerows(data_test)  # Data

    # return sigx, RF, CMOD, disp, steps


if __name__ == "__main__":
    # Generate training data
    E = Constant(15e3)
    nu = Constant(0.3)
    forward_model(E, nu)