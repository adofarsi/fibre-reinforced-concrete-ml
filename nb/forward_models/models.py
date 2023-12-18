import os
os.environ['OMP_NUM_THREADS'] = "1"
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

from .utilis import mesh2d_rectangle, cell_project, facet_project


def elastic_triaxial_synthetic_data(confining_pressures=[.5e6, 1e6, 2e6, 3e6]):
    
    pass

def cohesive_triaxial_curves(confining_pressure=1e6,
                             maximum_axial_strain=1e-4,
                             sample_width=.05,
                             sample_height=.1,
                             mesh_size=.0025,
                             youngs_modulus=20e9,
                             poissons_ratio=.25,
                             fracture_energy=3,
                             tensile_strength=4e6,
                             beta=2,
                             dg_penalty=1e10,
                             bc_penalty=1e11,
                             output_frequncy=0,
                             plot_mesh=False,
                             plot_output=False,
                             output_filename="cohesive_triaxial"):
    # Create mesh with Gmsh
    mesh = mesh2d_rectangle(sample_width, sample_height, mesh_size)
    if plot_mesh:
        fig1, ax1 = plt.subplots()
        fd.triplot(mesh, axes=ax1)
        plt.legend()
        plt.axis('equal')
        ax1.set_ylim(-sample_height*1.1/2, sample_height*1.1/2)
        ax1.set_xlim(-sample_width*1.1/2, sample_width*1.1/2)
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        plt.show()
    # Define function spaces
    V_u = fd.VectorFunctionSpace(mesh, "DG", 1)  # space for discontinuous displacement
    V_f = fd.FunctionSpace(mesh, "HDiv Trace", 0)  # space for facet-based internal variables (maximum gap)
    V_0 = fd.FunctionSpace(mesh, "DG", 0)  # space for piecewise constant material properties
    S1s_e = fd.FunctionSpace(mesh, "DG", 1)  # space for piecewise constant material properties
    dS = fd.dS
    # Define fields
    n = fd.FacetNormal(mesh)
    u = fd.Function(V_u, name="Displacement")
    u_, du = fd.TestFunction(V_u), fd.TrialFunction(V_u)
    g_max = fd.Function(V_f, name="Maximum opening")
    sigma_x = fd.Function(V_0, name="Sigma x")
    sigma_y = fd.Function(V_0, name="Sigma y")
    sigma_xy = fd.Function(V_0, name="Sigma xy")
    # Assigning material properties
    E_list = {1: youngs_modulus}
    E = cell_project(E_list, V_0)
    nu_list = {1: poissons_ratio}
    nu = cell_project(nu_list, V_0)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    Gc_list = {-1: fracture_energy}
    Gc = fd.avg(facet_project(Gc_list, V_f))
    sig_list = {-1: tensile_strength}
    sig_max = fd.avg(facet_project(sig_list, V_f))
    delta_0 = Gc/sig_max/fd.exp(1)
    beta = fd.Constant(beta)
    j_pen = fd.Constant(dg_penalty)
    pen = fd.Constant(bc_penalty)
    # Constitutive law
    def eps(v):
        return fd.sym(fd.grad(v))
    def sigma(v):
        d = mesh.geometric_dimension()
        return lmbda*fd.tr(eps(v))*fd.Identity(d) + 2*mu*eps(v)
    # Cohesive law
    def normal_opening(v, n):
        return fd.dot(v, n)
    def tangential_opening(v, n):
        return v - normal_opening(v, n)*n
    def effective_opening(v, n):
        return fd.sqrt(normal_opening(v, n)**2 + beta**2*tangential_opening(v, n)**2)
    def ppos(x):
        return (x+abs(x))/2.
    def nneg(x):
        return (x-abs(x))/2.
    def T(opening, g_max, n):
        return Gc/delta_0**2*fd.exp(-g_max/delta_0)*opening + j_pen * nneg(normal_opening(opening, n)) * n
    # Boundary conditions
    confinement = fd.Constant(confining_pressure)
    top_plate_u = fd.Constant(fd.as_vector([0, 0]))
    bottom_plate_u = fd.Constant(fd.as_vector([0, 0]))
    left_plate_u = fd.Constant(fd.as_vector([0, 0]))
    right_plate_u = fd.Constant(fd.as_vector([0, 0]))
    top_BC = pen * fd.dot(u_, (u - top_plate_u))*fd.ds(4)
    bottom_BC = pen * fd.dot(u_, (u - bottom_plate_u))*fd.ds(3)
    left_BC = pen * fd.dot(u_, (u - left_plate_u))*fd.ds(1)
    right_BC = pen * fd.dot(u_, (u - right_plate_u))*fd.ds(2)

    # Define variational problem
    F = fd.inner(sigma(u), eps(u_))*fd.dx +\
        fd.inner(T(fd.jump(u), fd.avg(g_max), n("-")), fd.jump(u_))*fd.dS +\
        top_BC + bottom_BC + left_BC + right_BC
    # Create a linear solver object
    nonlin_problem = fd.NonlinearVariationalProblem(F, u)
    nonlin_solver = fd.NonlinearVariationalSolver(nonlin_problem)
    # compute the size for the displacement increment of each iteration
    elastic_opening_displacement = tensile_strength / dg_penalty
    damage_opening_displacement = 2 * fracture_energy / tensile_strength
    critical_displacement = min(elastic_opening_displacement, damage_opening_displacement)
    iteration_displacement = critical_displacement / 10
    iteration_displacement_vector = fd.as_vector([0, iteration_displacement])
    # compute the maximum number of iterations
    maximum_displacement = maximum_axial_strain * sample_height
    maximum_number_of_iterations = int(maximum_displacement / iteration_displacement)
    # Write output pvd file
    if output_frequncy != 0:
        ffile = fd.File(f"output/{output_filename}.pvd")
    # Initialize lists for storing results
    axial_stress = [0.]
    axial_strain = [0.]
    radial_strain = [0.]
    radial_stress = [0.]
    # Start iterations
    for i in range(maximum_number_of_iterations*100):
        try:
            # Update boundary conditions
            if radial_stress[-1] < confining_pressure:
                left_plate_u.assign(left_plate_u + iteration_displacement_vector)
                right_plate_u.assign(right_plate_u - iteration_displacement_vector)
            if axial_stress[-1] < confining_pressure:
                top_plate_u.assign(top_plate_u - iteration_displacement_vector)
                bottom_plate_u.assign(bottom_plate_u + iteration_displacement_vector) 
            if radial_stress[-1] >= confining_pressure:    
                top_plate_u.assign(top_plate_u - iteration_displacement_vector)
                bottom_plate_u.assign(bottom_plate_u + iteration_displacement_vector)
                left_plate_u.assign(left_plate_u - iteration_displacement_vector)
                right_plate_u.assign(right_plate_u + iteration_displacement_vector)
            # Solve
            nonlin_solver.solve()
            # Update internal variables
            g_max.assign(facet_project(effective_opening(fd.jump(u), n("-")), V_f))
            # Update stress fields
            sigma_x.assign(fd.interpolate(sigma(u)[0,0], V_0))
            sigma_y.assign(fd.interpolate(sigma(u)[1,1], V_0))
            sigma_xy.assign(fd.interpolate(sigma(u)[1,1], V_0))
            # Append axial stress
            axial_stress.append(.5*(abs(fd.assemble(sigma(u)[1, 1]*fd.ds(3)))+abs(fd.assemble(sigma(u)[1, 1]*fd.ds(4))))/sample_width)
            radial_stress.append(.5*(abs(fd.assemble(sigma(u)[0, 0]*fd.ds(1)))+abs(fd.assemble(sigma(u)[0, 0]*fd.ds(2))))/sample_height)
            # Append axial and radial strain
            axial_strain.append(fd.assemble(u[1]*fd.ds(3) - u[1]*fd.ds(4))/(sample_height*sample_width))
            radial_strain.append(fd.assemble(u[0]*fd.ds(1) - u[0]*fd.ds(2))/(sample_width*sample_height))
            # Write output vtu file for iteration
            if output_frequncy!=0 and i%output_frequncy == 0:
                ffile.write(u, sigma_x, sigma_y, sigma_xy)
            # Check if maximum axial strain is reached
            # if axial_strain[-1] > maximum_axial_strain:
            #     if output_frequncy!=0 and i%output_frequncy != 0:
            #         # Write output vtu file for last iteration
            #         ffile.write(u, sigma_x, sigma_y, sigma_xy)
                break
        except Exception as error:
            print(error)
            break
    # Plot output
    if plot_output:
        plt.plot(axial_strain, 1e-6*np.array(axial_stress), label='Axial')
        plt.plot(radial_strain, 1e-6*np.array(axial_stress), label='Radial')
        plt.xlabel('Strain [-]')
        plt.ylabel('Stress [MPa]')
        plt.title(f'Confinement {confinement.values()[0]/1e6} MPa')
        plt.legend()
    
    
    return np.vstack([axial_stress, axial_strain, radial_strain])



def elastic_triaxial_curves(confining_pressure=1e6,
                             maximum_differential_stress=30e6,
                             number_of_iterations=100,
                             sample_width=.05,
                             sample_height=.1,
                             mesh_size=.0025,
                             youngs_modulus=20e9,
                             poissons_ratio=.25,
                             output_frequncy=0,
                             plot_mesh=False,
                             plot_output=False,
                             output_filename="elastic_triaxial"):
    # Create mesh with Gmsh
    mesh = mesh2d_rectangle(sample_width, sample_height, mesh_size)
    if plot_mesh:
        fig1, ax1 = plt.subplots()
        fd.triplot(mesh, axes=ax1)
        plt.legend()
        plt.axis('equal')
        ax1.set_ylim(-sample_height*1.1/2, sample_height*1.1/2)
        ax1.set_xlim(-sample_width*1.1/2, sample_width*1.1/2)
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        plt.show()
    # Define function spaces
    V_u = fd.VectorFunctionSpace(mesh, "CG", 1)  # space for discontinuous displacement
    V_0 = fd.FunctionSpace(mesh, "DG", 0)  # space for piecewise constant material properties
    n = fd.FacetNormal(mesh)
    # Define fields
    u = fd.Function(V_u, name="Displacement")
    u_, du = fd.TestFunction(V_u), fd.TrialFunction(V_u)
    sigma_x = fd.Function(V_0, name="Sigma x")
    sigma_y = fd.Function(V_0, name="Sigma y")
    sigma_xy = fd.Function(V_0, name="Sigma xy")
    # Assigning material properties
    E_list = {1: youngs_modulus}
    E = cell_project(E_list, V_0)
    nu_list = {1: poissons_ratio}
    nu = cell_project(nu_list, V_0)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
     # Constitutive law
    def eps(v):
        return fd.sym(fd.grad(v))
    def sigma(v):
        d = mesh.geometric_dimension()
        return lmbda*fd.tr(eps(v))*fd.Identity(d) + 2*mu*eps(v)
    # Boundary conditions
    # Dirichlet BC
    bottom_BC = fd.DirichletBC(V_u.sub(1), fd.Constant(0), 3)
    right_BC = fd.DirichletBC(V_u.sub(0), fd.Constant(0), 2)
    # Neumann BC
    confinement = fd.Constant(confining_pressure)
    axial_imposed_stress = fd.Constant(0)
    left_BC = (confinement) * fd.dot(n,u_)*fd.ds((1))
    top_BC = (confinement + axial_imposed_stress) * fd.dot(n,u_)*fd.ds((4))

    # Define variational problem
    F = fd.inner(sigma(u), eps(u_))*fd.dx +\
        top_BC + left_BC
 
    # Create a linear solver object
    nonlin_problem = fd.NonlinearVariationalProblem(F, u, bcs=[bottom_BC, right_BC])
    nonlin_solver = fd.NonlinearVariationalSolver(nonlin_problem)
    # compute the size for the stress increment of each iteration
    axial_stress_increment = maximum_differential_stress / number_of_iterations
 
    # Write output pvd file
    if output_frequncy != 0:
        ffile = fd.File(f"output/{output_filename}.pvd")
    # Initialize lists for storing results
    axial_stress = [0.]
    axial_strain = [0.]
    radial_strain = [0.]
    radial_stress = [0.]
    # Start iterations
    for i in range(number_of_iterations):
        # Update boundary conditions
        axial_imposed_stress.assign(axial_imposed_stress + axial_stress_increment)
        # Solve
        nonlin_solver.solve()
        # Update stress fields
        sigma_x.assign(fd.interpolate(sigma(u)[0,0], V_0))
        sigma_y.assign(fd.interpolate(sigma(u)[1,1], V_0))
        sigma_xy.assign(fd.interpolate(sigma(u)[1,1], V_0))
        # Append axial stress
        axial_stress.append(.5*(abs(fd.assemble(sigma(u)[1, 1]*fd.ds(3)))+abs(fd.assemble(sigma(u)[1, 1]*fd.ds(4))))/sample_width)
        radial_stress.append(.5*(abs(fd.assemble(sigma(u)[0, 0]*fd.ds(1)))+abs(fd.assemble(sigma(u)[0, 0]*fd.ds(2))))/sample_height)
        # Append axial and radial strain
        axial_strain.append(fd.assemble(u[1]*fd.ds(3) - u[1]*fd.ds(4))/(sample_height*sample_width))
        radial_strain.append(fd.assemble(u[0]*fd.ds(1) - u[0]*fd.ds(2))/(sample_width*sample_height))
        # Write output vtu file for iteration
        if output_frequncy!=0 and i%output_frequncy == 0:
            ffile.write(u, sigma_x, sigma_y, sigma_xy)

    # Plot output
    if plot_output:
        plt.plot(axial_strain, 1e-6*np.array(axial_stress), label='Axial')
        plt.plot(radial_strain, 1e-6*np.array(axial_stress), label='Radial')
        plt.xlabel('Strain [-]')
        plt.ylabel('Stress [MPa]')
        plt.title(f'Confinement {confinement.values()[0]/1e6} MPa')
        plt.legend()
    
    
    return np.vstack([axial_stress, axial_strain, radial_strain])